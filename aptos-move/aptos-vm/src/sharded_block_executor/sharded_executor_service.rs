// Copyright © Aptos Foundation
// SPDX-License-Identifier: Apache-2.0

use crate::{
    block_executor::BlockAptosVM,
    sharded_block_executor::{
        aggr_overridden_state_view::{AggregatorOverriddenStateView, TOTAL_SUPPLY_AGGR_BASE_VAL},
        coordinator_client::CoordinatorClient,
        counters::{
            SHARDED_BLOCK_EXECUTION_BY_ROUNDS_SECONDS, SHARDED_BLOCK_EXECUTOR_TXN_COUNT,
            SHARDED_EXECUTOR_SERVICE_SECONDS,
        },
        cross_shard_client::{CrossShardClient, CrossShardCommitReceiver, CrossShardCommitSender},
        cross_shard_state_view::CrossShardStateView,
        messages::CrossShardMsg,
        ExecutorShardCommand,
    },
};
use aptos_logger::{info, trace};
use aptos_types::{
    block_executor::{
        config::{BlockExecutorConfig, BlockExecutorLocalConfig},
        partitioner::{ShardId, SubBlock, SubBlocksForShard, TransactionWithDependencies},
    },
    state_store::StateView,
    transaction::{
        analyzed_transaction::AnalyzedTransaction,
        signature_verified_transaction::SignatureVerifiedTransaction, BlockOutput,
        TransactionOutput,
    },
};
use aptos_vm_logging::disable_speculative_logging;
use crossbeam_channel::{unbounded, Receiver, Sender};
use futures::{channel::oneshot, executor::block_on};
use move_core_types::vm_status::VMStatus;
use std::sync::{Arc, Mutex};
use std::thread;
use serde::{Deserialize, Serialize};
use aptos_mvhashmap::types::TxnIndex;

pub struct ShardedExecutorService<S: StateView + Sync + Send + 'static> {
    shard_id: ShardId,
    num_shards: usize,
    executor_thread_pool: Arc<rayon::ThreadPool>,
    coordinator_client: Arc<Mutex<dyn CoordinatorClient<S>>>,
    cross_shard_client: Arc<dyn CrossShardClient>,
}

impl<S: StateView + Sync + Send + 'static> ShardedExecutorService<S> {
    pub fn new(
        shard_id: ShardId,
        num_shards: usize,
        num_threads: usize,
        coordinator_client: Arc<Mutex<dyn CoordinatorClient<S>>>,
        cross_shard_client: Arc<dyn CrossShardClient>,
    ) -> Self {
        let executor_thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                // We need two extra threads for the cross-shard commit receiver and the thread
                // that is blocked on waiting for execute block to finish.
                .thread_name(move |i| format!("sharded-executor-shard-{}-{}", shard_id, i))
                .num_threads(num_threads + 2)
                .build()
                .unwrap(),
        );
        Self {
            shard_id,
            num_shards,
            executor_thread_pool,
            coordinator_client,
            cross_shard_client,
        }
    }

    fn execute_sub_block(
        &self,
        sub_block: SubBlock<AnalyzedTransaction>,
        round: usize,
        state_view: &S,
        config: BlockExecutorConfig,
        stream_result_tx: Sender<TransactionIdxAndOutput>,
    ) -> Result<Vec<TransactionOutput>, VMStatus> {
        disable_speculative_logging();
        trace!(
            "executing sub block for shard {} and round {}",
            self.shard_id,
            round
        );
        let cross_shard_commit_sender =
            CrossShardCommitSender::new(self.shard_id, self.cross_shard_client.clone(), &sub_block, stream_result_tx);
        Self::execute_transactions_with_dependencies(
            Some(self.shard_id),
            self.executor_thread_pool.clone(),
            sub_block.into_transactions_with_deps(),
            self.cross_shard_client.clone(),
            Some(cross_shard_commit_sender),
            round,
            state_view,
            config,
        )
    }

    pub fn execute_transactions_with_dependencies(
        shard_id: Option<ShardId>, // None means execution on global shard
        executor_thread_pool: Arc<rayon::ThreadPool>,
        transactions: Vec<TransactionWithDependencies<AnalyzedTransaction>>,
        cross_shard_client: Arc<dyn CrossShardClient>,
        cross_shard_commit_sender: Option<CrossShardCommitSender>,
        round: usize,
        state_view: &S,
        config: BlockExecutorConfig,
    ) -> Result<Vec<TransactionOutput>, VMStatus> {
        let (callback, callback_receiver) = oneshot::channel();

        let cross_shard_state_view = Arc::new(CrossShardStateView::create_cross_shard_state_view(
            state_view,
            &transactions,
        ));

        let cross_shard_state_view_clone = cross_shard_state_view.clone();
        let cross_shard_client_clone = cross_shard_client.clone();

        let aggr_overridden_state_view = Arc::new(AggregatorOverriddenStateView::new(
            cross_shard_state_view.as_ref(),
            TOTAL_SUPPLY_AGGR_BASE_VAL,
        ));

        let signature_verified_transactions: Vec<SignatureVerifiedTransaction> = transactions
            .into_iter()
            .map(|txn| txn.into_txn().into_txn())
            .collect();
        let executor_thread_pool_clone = executor_thread_pool.clone();

        executor_thread_pool.clone().scope(|s| {
            s.spawn(move |_| {
                CrossShardCommitReceiver::start(
                    cross_shard_state_view_clone,
                    cross_shard_client,
                    round,
                );
            });
            s.spawn(move |_| {
                let ret = BlockAptosVM::execute_block(
                    executor_thread_pool,
                    &signature_verified_transactions,
                    aggr_overridden_state_view.as_ref(),
                    config,
                    cross_shard_commit_sender,
                )
                .map(BlockOutput::into_transaction_outputs_forced);
                if let Some(shard_id) = shard_id {
                    trace!(
                        "executed sub block for shard {} and round {}",
                        shard_id,
                        round
                    );
                    // Send a self message to stop the cross-shard commit receiver.
                    cross_shard_client_clone.send_cross_shard_msg(
                        shard_id,
                        round,
                        CrossShardMsg::StopMsg,
                    );
                } else {
                    trace!("executed block for global shard and round {}", round);
                    // Send a self message to stop the cross-shard commit receiver.
                    cross_shard_client_clone.send_global_msg(CrossShardMsg::StopMsg);
                }
                callback.send(ret).unwrap();
                executor_thread_pool_clone.spawn(move || {
                    // Explicit async drop
                    drop(signature_verified_transactions);
                });
            });
        });

        block_on(callback_receiver).unwrap()
    }

    fn execute_block(
        &self,
        transactions: SubBlocksForShard<AnalyzedTransaction>,
        state_view: &S,
        config: BlockExecutorConfig,
        stream_result_tx: Sender<TransactionIdxAndOutput>,
    ) -> Result<Vec<Vec<TransactionOutput>>, VMStatus> {
        let mut result = vec![];
        for (round, sub_block) in transactions.into_sub_blocks().into_iter().enumerate() {
            let _timer = SHARDED_BLOCK_EXECUTION_BY_ROUNDS_SECONDS
                .with_label_values(&[&self.shard_id.to_string(), &round.to_string()])
                .start_timer();
            SHARDED_BLOCK_EXECUTOR_TXN_COUNT
                .with_label_values(&[&self.shard_id.to_string(), &round.to_string()])
                .observe(sub_block.transactions.len() as f64);
            info!(
                "executing sub block for shard {} and round {}, number of txns {}",
                self.shard_id,
                round,
                sub_block.transactions.len()
            );
            result.push(self.execute_sub_block(
                sub_block,
                round,
                state_view,
                config.clone(),
                stream_result_tx.clone(),
            )?);
            trace!(
                "Finished executing sub block for shard {} and round {}",
                self.shard_id,
                round
            );
        }
        Ok(result)
    }

    pub fn start(&mut self) {
        trace!(
            "Shard starting, shard_id={}, num_shards={}.",
            self.shard_id,
            self.num_shards
        );

        let mut num_txns = 0;
        loop {
            let command = self.coordinator_client.lock().unwrap().receive_execute_command();
            match command {
                ExecutorShardCommand::ExecuteSubBlocks(
                    state_view,
                    transactions,
                    concurrency_level_per_shard,
                    onchain_config,
                ) => {
                    num_txns += transactions.num_txns();
                    let (stream_results_tx, stream_results_rx) = unbounded();
                    let coordinator_client_clone = self.coordinator_client.clone();
                    let stream_results_thread = thread::spawn(move || {
                        loop {
                            let txn_idx_output: TransactionIdxAndOutput = stream_results_rx.recv().unwrap();
                            if txn_idx_output.txn_idx == u32::MAX {
                                break;
                            }
                            coordinator_client_clone.lock().unwrap().send_single_execution_result(txn_idx_output);
                        }
                    });

                    trace!(
                        "Shard {} received ExecuteBlock command of block size {} ",
                        self.shard_id,
                        num_txns
                    );
                    let exe_timer = SHARDED_EXECUTOR_SERVICE_SECONDS
                        .with_label_values(&[&self.shard_id.to_string(), "execute_block"])
                        .start_timer();
                    let ret = self.execute_block(
                        transactions,
                        state_view.as_ref(),
                        BlockExecutorConfig {
                            local: BlockExecutorLocalConfig {
                                concurrency_level: concurrency_level_per_shard,
                                allow_fallback: true,
                                discard_failed_blocks: false,
                            },
                            onchain: onchain_config,
                        },
                        stream_results_tx.clone(),
                    );
                    drop(state_view);
                    drop(exe_timer);

                    stream_results_tx.send(TransactionIdxAndOutput {
                        txn_idx: u32::MAX,
                        txn_output: TransactionOutput::default(),
                    }).unwrap();
                    stream_results_thread.join().unwrap();
                    /*let _result_tx_timer = SHARDED_EXECUTOR_SERVICE_SECONDS
                        .with_label_values(&[&self.shard_id.to_string(), "result_tx"])
                        .start_timer();
                    self.coordinator_client.lock().unwrap().send_execution_result(ret);*/
                },
                ExecutorShardCommand::Stop => {
                    break;
                },
            }
        }

        let exe_time = SHARDED_EXECUTOR_SERVICE_SECONDS
            .get_metric_with_label_values(&[&self.shard_id.to_string(), "execute_block"])
            .unwrap()
            .get_sample_sum();
        info!(
            "Shard {} is shutting down; On shard execution tps {} txns/s ({} txns / {} s)",
            self.shard_id,
            (num_txns as f64 / exe_time),
            num_txns,
            exe_time
        );
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TransactionIdxAndOutput {
    pub txn_idx: TxnIndex,
    pub txn_output: TransactionOutput,
}