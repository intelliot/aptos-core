// Copyright © Aptos Foundation
// Parts of the project are originally copyright © Meta Platforms, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    counters, pipeline::hashable::Hashable, state_replication::StateComputerCommitCallBackType,
};
use anyhow::anyhow;
use aptos_consensus_types::{
    common::{Author, Round},
    pipeline::commit_vote::CommitVote,
    pipelined_block::PipelinedBlock,
};
use aptos_crypto::{bls12381, HashValue};
use aptos_executor_types::ExecutorResult;
use aptos_logger::prelude::*;
use aptos_reliable_broadcast::DropGuard;
use aptos_types::{
    aggregate_signature::PartialSignatures,
    block_info::BlockInfo,
    epoch_state::EpochState,
    ledger_info::{
        LedgerInfo, LedgerInfoWithMixedSignatures, LedgerInfoWithSignatures, VerificationStatus,
    },
};
use futures::future::BoxFuture;
use itertools::zip_eq;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use tokio::time::Instant;

fn generate_commit_ledger_info(
    commit_info: &BlockInfo,
    ordered_proof: &LedgerInfoWithSignatures,
    order_vote_enabled: bool,
) -> LedgerInfo {
    LedgerInfo::new(
        commit_info.clone(),
        if order_vote_enabled {
            HashValue::zero()
        } else {
            ordered_proof.ledger_info().consensus_data_hash()
        },
    )
}

fn verify_signatures(
    unverified_signatures: PartialSignatures,
    epoch_state: Arc<EpochState>,
    commit_ledger_info: &LedgerInfo,
) -> PartialSignatures {
    // Returns a valid partial signature from a set of unverified signatures.
    // TODO: Validating individual signatures in expensive. Replace this with optimistic signature
    // verification for BLS. Here, we can implement a tree-based batch verification technique that
    // filters out invalid signature shares much faster when there are only a few of them
    // (e.g., [LM07]: Finding Invalid Signatures in Pairing-Based Batches,
    // by Law, Laurie and Matt, Brian J., in Cryptography and Coding, 2007).
    if let Ok(aggregated_signature) = epoch_state
        .verifier
        .aggregate_signatures(&unverified_signatures)
    {
        if epoch_state
            .verifier
            .verify_multi_signatures(commit_ledger_info, &aggregated_signature)
            .is_ok()
        {
            return unverified_signatures;
        }
    }

    PartialSignatures::new(
        unverified_signatures
            .signatures()
            .clone()
            .into_par_iter()
            .flat_map(|(account_address, signature)| {
                let _timer = counters::VERIFY_MSG
                    .with_label_values(&["commit_votes_from_ordered_bufer_item"])
                    .start_timer();
                if epoch_state
                    .verifier
                    .verify(account_address, commit_ledger_info, &signature)
                    .is_ok()
                {
                    return Some((account_address, signature));
                }
                None
            })
            .collect(),
    )
}

fn generate_executed_item_from_ordered(
    commit_info: BlockInfo,
    executed_blocks: Vec<PipelinedBlock>,
    verified_signatures: PartialSignatures,
    callback: StateComputerCommitCallBackType,
    ordered_proof: LedgerInfoWithSignatures,
    order_vote_enabled: bool,
) -> BufferItem {
    debug!("{} advance to executed from ordered", commit_info);
    let mut partial_commit_proof = LedgerInfoWithMixedSignatures::new(generate_commit_ledger_info(
        &commit_info,
        &ordered_proof,
        order_vote_enabled,
    ));
    for (author, sig) in verified_signatures.signatures() {
        partial_commit_proof.add_signature(*author, sig.clone(), VerificationStatus::Verified);
    }
    BufferItem::Executed(Box::new(ExecutedItem {
        executed_blocks,
        partial_commit_proof,
        callback,
        commit_info,
        ordered_proof,
    }))
}

fn aggregate_commit_proof(
    commit_ledger_info: &LedgerInfo,
    verified_signatures: &PartialSignatures,
    epoch_state: Arc<EpochState>,
) -> LedgerInfoWithSignatures {
    let aggregated_sig = epoch_state
        .verifier
        .aggregate_signatures(verified_signatures)
        .expect("Failed to generate aggregated signature");
    LedgerInfoWithSignatures::new(commit_ledger_info.clone(), aggregated_sig)
}

// we differentiate buffer items at different stages
// for better code readability
pub struct OrderedItem {
    pub unverified_signatures: PartialSignatures,
    // This can happen in the fast forward sync path, where we can receive the commit proof
    // from peers.
    pub commit_proof: Option<LedgerInfoWithSignatures>,
    pub callback: StateComputerCommitCallBackType,
    pub ordered_blocks: Vec<PipelinedBlock>,
    pub ordered_proof: LedgerInfoWithSignatures,
}

pub struct ExecutedItem {
    pub executed_blocks: Vec<PipelinedBlock>,
    pub partial_commit_proof: LedgerInfoWithMixedSignatures,
    pub callback: StateComputerCommitCallBackType,
    pub commit_info: BlockInfo,
    pub ordered_proof: LedgerInfoWithSignatures,
}

pub struct SignedItem {
    pub executed_blocks: Vec<PipelinedBlock>,
    pub partial_commit_proof: LedgerInfoWithMixedSignatures,
    pub callback: StateComputerCommitCallBackType,
    pub commit_vote: CommitVote,
    pub rb_handle: Option<(Instant, DropGuard)>,
}

pub struct AggregatedItem {
    pub executed_blocks: Vec<PipelinedBlock>,
    pub commit_proof: LedgerInfoWithSignatures,
    pub callback: StateComputerCommitCallBackType,
}

pub enum BufferItem {
    Ordered(Box<OrderedItem>),
    Executed(Box<ExecutedItem>),
    Signed(Box<SignedItem>),
    Aggregated(Box<AggregatedItem>),
}

impl Hashable for BufferItem {
    fn hash(&self) -> HashValue {
        self.block_id()
    }
}

pub type ExecutionFut = BoxFuture<'static, ExecutorResult<Vec<PipelinedBlock>>>;

impl BufferItem {
    pub fn new_ordered(
        ordered_blocks: Vec<PipelinedBlock>,
        ordered_proof: LedgerInfoWithSignatures,
        callback: StateComputerCommitCallBackType,
    ) -> Self {
        Self::Ordered(Box::new(OrderedItem {
            unverified_signatures: PartialSignatures::empty(),
            commit_proof: None,
            callback,
            ordered_blocks,
            ordered_proof,
        }))
    }

    // pipeline functions
    pub fn advance_to_executed_or_aggregated(
        self,
        executed_blocks: Vec<PipelinedBlock>,
        epoch_state: Arc<EpochState>,
        epoch_end_timestamp: Option<u64>,
        order_vote_enabled: bool,
    ) -> Self {
        match self {
            Self::Ordered(ordered_item) => {
                let OrderedItem {
                    ordered_blocks,
                    commit_proof,
                    unverified_signatures,
                    callback,
                    ordered_proof,
                } = *ordered_item;
                for (b1, b2) in zip_eq(ordered_blocks.iter(), executed_blocks.iter()) {
                    assert_eq!(b1.id(), b2.id());
                }
                let mut commit_info = executed_blocks
                    .last()
                    .expect("execute_blocks should not be empty!")
                    .block_info();
                match epoch_end_timestamp {
                    Some(timestamp) if commit_info.timestamp_usecs() != timestamp => {
                        assert!(executed_blocks
                            .last()
                            .expect("")
                            .is_reconfiguration_suffix());
                        commit_info.change_timestamp(timestamp);
                    },
                    _ => (),
                }
                if let Some(commit_proof) = commit_proof {
                    // We have already received the commit proof in fast forward sync path,
                    // we can just use that proof and proceed to aggregated
                    assert_eq!(commit_proof.commit_info().clone(), commit_info);
                    debug!(
                        "{} advance to aggregated from ordered",
                        commit_proof.commit_info()
                    );
                    Self::Aggregated(Box::new(AggregatedItem {
                        executed_blocks,
                        commit_proof,
                        callback,
                    }))
                } else {
                    let commit_ledger_info = generate_commit_ledger_info(
                        &commit_info,
                        &ordered_proof,
                        order_vote_enabled,
                    );

                    let verified_signatures = verify_signatures(
                        unverified_signatures,
                        epoch_state.clone(),
                        &commit_ledger_info,
                    );
                    if (epoch_state
                        .verifier
                        .check_voting_power(verified_signatures.signatures().keys(), true))
                    .is_ok()
                    {
                        let commit_proof = aggregate_commit_proof(
                            &commit_ledger_info,
                            &verified_signatures,
                            epoch_state,
                        );
                        debug!(
                            "{} advance to aggregated from ordered",
                            commit_proof.commit_info()
                        );
                        Self::Aggregated(Box::new(AggregatedItem {
                            executed_blocks,
                            commit_proof,
                            callback,
                        }))
                    } else {
                        generate_executed_item_from_ordered(
                            commit_info,
                            executed_blocks,
                            verified_signatures,
                            callback,
                            ordered_proof,
                            order_vote_enabled,
                        )
                    }
                }
            },
            _ => {
                panic!("Only ordered blocks can advance to executed blocks.")
            },
        }
    }

    pub fn advance_to_signed(self, author: Author, signature: bls12381::Signature) -> Self {
        match self {
            Self::Executed(executed_item) => {
                let ExecutedItem {
                    executed_blocks,
                    callback,
                    partial_commit_proof,
                    ..
                } = *executed_item;

                // we don't add the signature here, it'll be added when receiving the commit vote from self
                let commit_vote = CommitVote::new_with_signature(
                    author,
                    partial_commit_proof.ledger_info().clone(),
                    signature,
                );
                debug!("{} advance to signed", partial_commit_proof.commit_info());

                Self::Signed(Box::new(SignedItem {
                    executed_blocks,
                    callback,
                    partial_commit_proof,
                    commit_vote,
                    rb_handle: None,
                }))
            },
            _ => {
                panic!("Only executed buffer items can advance to signed blocks.")
            },
        }
    }

    /// this function assumes block id matches and the validity of ledger_info and that it has the voting power
    /// it returns an updated item
    pub fn try_advance_to_aggregated_with_ledger_info(
        self,
        commit_proof: LedgerInfoWithSignatures,
    ) -> Self {
        match self {
            Self::Signed(signed_item) => {
                let SignedItem {
                    executed_blocks,
                    callback,
                    partial_commit_proof: local_commit_proof,
                    ..
                } = *signed_item;
                assert_eq!(local_commit_proof.commit_info(), commit_proof.commit_info(),);
                debug!(
                    "{} advance to aggregated with commit decision",
                    commit_proof.commit_info()
                );
                Self::Aggregated(Box::new(AggregatedItem {
                    executed_blocks,
                    callback,
                    commit_proof,
                }))
            },
            Self::Executed(executed_item) => {
                let ExecutedItem {
                    executed_blocks,
                    callback,
                    commit_info,
                    ..
                } = *executed_item;
                assert_eq!(commit_info, *commit_proof.commit_info());
                debug!(
                    "{} advance to aggregated with commit decision",
                    commit_proof.commit_info()
                );
                Self::Aggregated(Box::new(AggregatedItem {
                    executed_blocks,
                    callback,
                    commit_proof,
                }))
            },
            Self::Ordered(ordered_item) => {
                let ordered = *ordered_item;
                assert!(ordered
                    .ordered_proof
                    .commit_info()
                    .match_ordered_only(commit_proof.commit_info()));
                // can't aggregate it without execution, only store the signatures
                debug!(
                    "{} received commit decision in ordered stage",
                    commit_proof.commit_info()
                );
                Self::Ordered(Box::new(OrderedItem {
                    commit_proof: Some(commit_proof),
                    ..ordered
                }))
            },
            Self::Aggregated(_) => {
                unreachable!("Found aggregated buffer item but any aggregated buffer item should get dequeued right away.");
            },
        }
    }

    pub fn try_advance_to_aggregated(self, epoch_state: Arc<EpochState>) -> Self {
        match self {
            Self::Signed(signed_item) => {
                if signed_item
                    .partial_commit_proof
                    .check_voting_power(&epoch_state.verifier)
                    .is_ok()
                {
                    let _time = counters::VERIFY_MSG
                        .with_label_values(&["commit_vote_aggregate_and_verify"])
                        .start_timer();
                    if let Ok(commit_proof) = signed_item
                        .partial_commit_proof
                        .clone()
                        .aggregate_and_verify(epoch_state.clone())
                    {
                        return Self::Aggregated(Box::new(AggregatedItem {
                            executed_blocks: signed_item.executed_blocks,
                            commit_proof,
                            callback: signed_item.callback,
                        }));
                    }
                }
                Self::Signed(signed_item)
            },
            Self::Executed(executed_item) => {
                if executed_item
                    .partial_commit_proof
                    .check_voting_power(&epoch_state.verifier)
                    .is_ok()
                {
                    let _time = counters::VERIFY_MSG
                        .with_label_values(&["commit_vote_aggregate_and_verify"])
                        .start_timer();

                    if let Ok(commit_proof) = executed_item
                        .partial_commit_proof
                        .clone()
                        .aggregate_and_verify(epoch_state.clone())
                    {
                        return Self::Aggregated(Box::new(AggregatedItem {
                            executed_blocks: executed_item.executed_blocks,
                            commit_proof,
                            callback: executed_item.callback,
                        }));
                    }
                }
                Self::Executed(executed_item)
            },
            _ => self,
        }
    }

    // generic functions
    pub fn get_blocks(&self) -> &Vec<PipelinedBlock> {
        match self {
            Self::Ordered(ordered) => &ordered.ordered_blocks,
            Self::Executed(executed) => &executed.executed_blocks,
            Self::Signed(signed) => &signed.executed_blocks,
            Self::Aggregated(aggregated) => &aggregated.executed_blocks,
        }
    }

    pub fn block_id(&self) -> HashValue {
        self.get_blocks()
            .last()
            .expect("Vec<PipelinedBlock> should not be empty")
            .id()
    }

    pub fn round(&self) -> Round {
        self.get_blocks()
            .last()
            .expect("Vec<PipelinedBlock> should not be empty")
            .round()
    }

    pub fn add_signature_if_matched(
        &mut self,
        vote: CommitVote,
        verification_status: VerificationStatus,
    ) -> anyhow::Result<()> {
        let target_commit_info = vote.commit_info();
        let author = vote.author();
        let signature = vote.signature().clone();
        match self {
            Self::Ordered(ordered) => {
                if ordered
                    .ordered_proof
                    .commit_info()
                    .match_ordered_only(target_commit_info)
                {
                    // we optimistically assume the vote will be valid in the future.
                    // when advancing to executed item, we will check if the sigs are valid.
                    // each author at most stores a single sig for each item,
                    // so an adversary will not be able to flood our memory.
                    ordered
                        .unverified_signatures
                        .add_signature(author, signature);
                    return Ok(());
                }
            },
            Self::Executed(executed) => {
                if executed.commit_info == *target_commit_info {
                    executed.partial_commit_proof.add_signature(
                        author,
                        signature,
                        verification_status,
                    );
                    return Ok(());
                }
            },
            Self::Signed(signed) => {
                if signed.partial_commit_proof.commit_info() == target_commit_info {
                    signed.partial_commit_proof.add_signature(
                        author,
                        signature,
                        verification_status,
                    );
                    return Ok(());
                }
            },
            Self::Aggregated(aggregated) => {
                // we do not need to do anything for aggregated
                // but return true is helpful to stop the outer loop early
                if aggregated.commit_proof.commit_info() == target_commit_info {
                    return Ok(());
                }
            },
        }
        Err(anyhow!("Inconsistent commit info."))
    }

    pub fn is_ordered(&self) -> bool {
        matches!(self, Self::Ordered(_))
    }

    pub fn is_executed(&self) -> bool {
        matches!(self, Self::Executed(_))
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Self::Signed(_))
    }

    pub fn is_aggregated(&self) -> bool {
        matches!(self, Self::Aggregated(_))
    }

    pub fn unwrap_signed_mut(&mut self) -> &mut SignedItem {
        match self {
            BufferItem::Signed(item) => item.as_mut(),
            _ => panic!("Not signed item"),
        }
    }

    pub fn unwrap_executed_ref(&self) -> &ExecutedItem {
        match self {
            BufferItem::Executed(item) => item.as_ref(),
            _ => panic!("Not executed item"),
        }
    }

    pub fn unwrap_aggregated(self) -> AggregatedItem {
        match self {
            BufferItem::Aggregated(item) => *item,
            _ => panic!("Not aggregated item"),
        }
    }
}
