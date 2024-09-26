// Copyright (c) Aptos Foundation
// SPDX-License-Identifier: Apache-2.0

// Copyright © Aptos Foundation

use crate::{BlockPartitioner, PartitionerConfig};
use aptos_types::{
    block_executor::partitioner::{
        PartitionV3, PartitionedTransactions, PartitionedTransactionsV3,
    },
    state_store::state_key::StateKey,
    transaction::analyzed_transaction::AnalyzedTransaction,
};
use std::collections::{HashMap, HashSet};
use aptos_logger::info;

/// A partitioner that does not reorder and assign txns to shards in a round-robin way.
/// Only for testing the correctness or sharded execution V3.
#[derive(Default)]
pub struct V3NaivePartitioner {
    pub print_debug_stats: bool,
}

impl BlockPartitioner for V3NaivePartitioner {
    fn partition(
        &self,
        transactions: Vec<AnalyzedTransaction>,
        num_shards: usize,
    ) -> PartitionedTransactions {
        let shard_idxs = (0..transactions.len()).map(|i| i % num_shards).collect();
        PartitionedTransactions::V3(build_partitioning_result(num_shards, transactions, shard_idxs, self.print_debug_stats))
        /*let shard_idx_of_txn = |txn_idx: u32| txn_idx as usize % num_shards; // Naive Round-Robin.
        let mut partitions = vec![PartitionV3::default(); num_shards];
        let mut owners_by_key: HashMap<StateKey, u32> = HashMap::new();
        for (cur_txn_idx, transaction) in transactions.into_iter().enumerate() {
            let cur_shard_idx = shard_idx_of_txn(cur_txn_idx as u32);

            // Find remote dependencies with reads + writes.
            for loc in transaction
                .read_hints
                .iter()
                .chain(transaction.write_hints.iter())
            {
                if let Some(owner_txn_idx) = owners_by_key.get(loc.state_key()) {
                    let owner_shard_idx = shard_idx_of_txn(*owner_txn_idx);
                    if owner_shard_idx == cur_shard_idx {
                        continue;
                    }
                    partitions[owner_shard_idx]
                        .insert_follower_shard(*owner_txn_idx, cur_shard_idx);
                    partitions[cur_shard_idx]
                        .insert_remote_dependency(*owner_txn_idx, loc.state_key().clone());
                }
            }

            // Update owner table with writes.
            for loc in transaction.write_hints.iter() {
                owners_by_key.insert(loc.state_key().clone(), cur_txn_idx as u32);
            }

            partitions[cur_shard_idx].append_txn(cur_txn_idx as u32, transaction);
        }

        let global_idx_lists_by_shard = partitions.iter().map(|p| p.global_idxs.clone()).collect();

        PartitionedTransactions::V3(PartitionedTransactionsV3 {
            block_id: [0; 32],
            partitions,
            global_idx_sets_by_shard: global_idx_lists_by_shard,
        })*/
    }
}

#[derive(Debug, Default)]
pub struct V3NaivePartitionerConfig {}

impl PartitionerConfig for V3NaivePartitionerConfig {
    fn build(&self) -> Box<dyn BlockPartitioner> {
        Box::new(V3NaivePartitioner::default())
    }
}

pub fn build_partitioning_result(num_shards: usize, transactions: Vec<AnalyzedTransaction>, shard_idxs: Vec<usize>, print_debug_stats: bool) -> PartitionedTransactionsV3 {
    let mut partitions = vec![PartitionV3::default(); num_shards];
    let mut owners_by_key: HashMap<StateKey, u32> = HashMap::new();

    // Track remote dependencies:
    // shard_idx -> self_local_idx -> (owner_txn_local_idx, shard_idx)
    let mut remote_dependency_positions: Vec<HashMap<usize, HashSet<(usize, usize)>>> = vec![HashMap::new(); num_shards];

    for (cur_txn_idx, transaction) in transactions.into_iter().enumerate() {
        let cur_shard_idx = shard_idxs[cur_txn_idx];

        // Find remote dependencies with reads + writes.
        for loc in transaction
            .read_hints
            .iter()
            .chain(transaction.write_hints.iter())
        {
            if let Some(owner_txn_idx) = owners_by_key.get(loc.state_key()) {
                let owner_shard_idx = shard_idxs[*owner_txn_idx as usize];
                if owner_shard_idx == cur_shard_idx {
                    continue;
                }
                partitions[owner_shard_idx]
                    .insert_follower_shard(*owner_txn_idx, cur_shard_idx);
                partitions[cur_shard_idx]
                    .insert_remote_dependency(*owner_txn_idx, loc.state_key().clone());

                // Track remote dependency positions
                if print_debug_stats {
                    let current_txn_local_idx = partitions[cur_shard_idx].num_txns();
                    let owner_txn_local_idx = partitions[owner_shard_idx].local_idx_by_global.get(owner_txn_idx).unwrap();
                    remote_dependency_positions[cur_shard_idx].entry(current_txn_local_idx).or_insert(HashSet::new()).insert((*owner_txn_local_idx, owner_shard_idx));
                }
            }
        }

        // Update owner table with writes.
        for loc in transaction.write_hints.iter() {
            owners_by_key.insert(loc.state_key().clone(), cur_txn_idx as u32);
        }

        partitions[cur_shard_idx].append_txn(cur_txn_idx as u32, transaction);
    }

    let global_idx_lists_by_shard = partitions.iter().map(|p| p.global_idxs.clone()).collect();

    if print_debug_stats {
        partitioning_stats(&partitions, remote_dependency_positions);
    }

    PartitionedTransactionsV3 {
        block_id: [0; 32],
        partitions,
        global_idx_sets_by_shard: global_idx_lists_by_shard,
    }
}

fn partitioning_stats(partitions: &Vec<PartitionV3>, remote_dependency_positions: Vec<HashMap<usize, HashSet<(usize, usize)>>>) {
    let mut overall_remote_deps = 0;
    let mut overall_sum_remote_dep_pos: f64 = 0.0;
    let mut overall_min_remote_dep_pos: usize = std::usize::MAX;
    let mut overall_max_remote_dep_pos: usize = 0;

    let mut overall_owner_txns = 0;
    let mut overall_sum_owner_txn_pos: f64 = 0.0;
    let mut overall_min_owner_txn_pos: usize = std::usize::MAX;
    let mut overall_max_owner_txn_pos: usize = 0;

    let mut overall_dep_to_owner_pos_diff: i64 = 0;

    for (shard_idx, partition) in partitions.iter().enumerate() {
        let shard_size = partition.num_txns();
        if shard_size == 0 {
            info!("Shard {}: Empty", shard_idx);
            continue;
        }
        let remote_deps_size = remote_dependency_positions[shard_idx].len();
        let mut sum_remote_dep_pos: usize = 0;
        let mut min_remote_dep_pos: usize = std::usize::MAX;
        let mut max_remote_dep_pos: usize = 0;

        let mut dep_to_owner_pos_diff: i64 = 0;

        for (dep_idx, entry) in remote_dependency_positions[shard_idx].iter() {
            for ((owner_idx, owner_shard_idx)) in entry.iter() {
                sum_remote_dep_pos += dep_idx;
                min_remote_dep_pos = std::cmp::min(min_remote_dep_pos, *dep_idx);
                max_remote_dep_pos = std::cmp::max(max_remote_dep_pos, *dep_idx);
                dep_to_owner_pos_diff += (*dep_idx as i64 - *owner_idx as i64);
            }
        }

        let avg_remote_dep_pos = if remote_deps_size == 0 {
            0.0
        } else {
            sum_remote_dep_pos as f64 / (remote_deps_size * shard_size) as f64
        };
        info!("Shard {}, Size {}", shard_idx, shard_size);
        info!("                    [dep txns] Num remote deps: {}, Avg dep pos: {:.2}, (min: {}, max: {}), Avg dep_to_owner_pos_diff: {:.2}",
                 remote_deps_size, avg_remote_dep_pos, min_remote_dep_pos, max_remote_dep_pos, dep_to_owner_pos_diff as f64 / remote_deps_size as f64);

        overall_remote_deps += remote_deps_size;
        overall_sum_remote_dep_pos += (avg_remote_dep_pos * remote_deps_size as f64);
        overall_min_remote_dep_pos = std::cmp::min(overall_min_remote_dep_pos, min_remote_dep_pos);
        overall_max_remote_dep_pos = std::cmp::max(overall_max_remote_dep_pos, max_remote_dep_pos);

        overall_dep_to_owner_pos_diff += dep_to_owner_pos_diff;

        let mut num_owner_txns = 0;
        let mut sum_owner_txn_pos: usize = 0;
        let mut min_owner_txn_pos: usize = std::usize::MAX;
        let mut max_owner_txn_pos: usize = 0;
        for (pos, hash_set) in partition.follower_shard_sets.iter().enumerate() {
            if hash_set.len() == 0 {
                continue;
            }
            num_owner_txns += hash_set.len();
            sum_owner_txn_pos += pos;
            min_owner_txn_pos = std::cmp::min(min_owner_txn_pos, pos);
            max_owner_txn_pos = std::cmp::max(max_owner_txn_pos, pos);
        }
        let avg_owner_txn_pos = if num_owner_txns == 0 {
            0.0
        } else {
            sum_owner_txn_pos as f64 / (num_owner_txns * shard_size) as f64
        };
        info!("                    [owner txns] Num owner txns: {}, Avg owner txn pos: {:.2}, (min: {}, max: {})",
                 num_owner_txns, avg_owner_txn_pos, min_owner_txn_pos, max_owner_txn_pos);

        overall_owner_txns += num_owner_txns;
        overall_sum_owner_txn_pos += (avg_owner_txn_pos * num_owner_txns as f64);
        overall_min_owner_txn_pos = std::cmp::min(overall_min_owner_txn_pos, min_owner_txn_pos);
        overall_max_owner_txn_pos = std::cmp::max(overall_max_owner_txn_pos, max_owner_txn_pos);
    }
    info!("[Overall dep txns stats]: Num remote deps: {}, Avg dep pos: {:.2} (min: {}, max: {}), Avg dep_to_owner_pos_diff: {:.2}",
             overall_remote_deps, overall_sum_remote_dep_pos / overall_remote_deps as f64, overall_min_remote_dep_pos, overall_max_remote_dep_pos, overall_dep_to_owner_pos_diff as f64 / remote_dependency_positions.iter().map(|p| p.len()).sum::<usize>() as f64);
    info!("[Overall owner txns stats]: Num owner txns: {}, Avg owner txn pos: {:.2}, (min: {}, max: {})",
             overall_owner_txns, overall_sum_owner_txn_pos / overall_owner_txns as f64, overall_min_owner_txn_pos, overall_max_owner_txn_pos);
}
