// Copyright © Aptos Foundation
// SPDX-License-Identifier: Apache-2.0

use crate::{
    logging::{LogEvent, LogSchema},
    monitor,
    network::QuorumStoreSender,
    quorum_store::{
        batch_generator::BatchGeneratorCommand, batch_store::BatchReader, counters, utils::Timeouts,
    },
};
use aptos_consensus_types::proof_of_store::{
    BatchInfo, ProofCache, ProofOfStore, SignedBatchInfo, SignedBatchInfoError, SignedBatchInfoMsg,
};
use aptos_logger::prelude::*;
use aptos_types::{
    aggregate_signature::PartialSignatures, epoch_state::EpochState,
    validator_verifier::ValidatorVerifier, PeerId,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc::Receiver, oneshot as TokioOneshot},
    time,
};

#[derive(Debug)]
pub(crate) enum ProofCoordinatorCommand {
    // The bool indicates whether the signed batch info message is already verified.
    // If the message is not verified, the coordinator is exepected to verify the message.
    AppendSignature((SignedBatchInfoMsg, bool)),
    CommitNotification(Vec<BatchInfo>),
    Shutdown(TokioOneshot::Sender<()>),
}

struct IncrementalProofState {
    info: BatchInfo,
    unverified_signatures: PartialSignatures,
    verified_signatures: PartialSignatures,
    self_voted: bool,
    completed: bool,
}

impl IncrementalProofState {
    fn new(info: BatchInfo) -> Self {
        Self {
            info,
            unverified_signatures: PartialSignatures::empty(),
            verified_signatures: PartialSignatures::empty(),
            self_voted: false,
            completed: false,
        }
    }

    pub fn all_voters(&self) -> Vec<PeerId> {
        self.verified_signatures
            .signatures()
            .keys()
            .chain(self.unverified_signatures.signatures().keys())
            .cloned()
            .collect()
    }

    pub fn voter_count(&self) -> u64 {
        self.verified_signatures.signatures().len() as u64
            + self.unverified_signatures.signatures().len() as u64
    }

    pub fn aggregate_voting_power(&self, verifier: &ValidatorVerifier) -> u64 {
        verifier
            .check_voting_power(self.all_voters().iter(), true)
            .unwrap_or(0) as u64
    }

    fn add_signature(
        &mut self,
        signed_batch_info: &SignedBatchInfo,
        epoch_state: Arc<EpochState>,
        verified: bool,
    ) -> Result<(), SignedBatchInfoError> {
        if signed_batch_info.batch_info() != &self.info {
            return Err(SignedBatchInfoError::WrongInfo((
                signed_batch_info.batch_id().id,
                self.info.batch_id().id,
            )));
        }

        match epoch_state
            .verifier
            .get_voting_power(&signed_batch_info.signer())
        {
            Some(_voting_power) => {
                let signer = signed_batch_info.signer();
                if verified {
                    self.verified_signatures
                        .add_signature(signer, signed_batch_info.signature().clone());
                    self.unverified_signatures.remove_signature(signer);
                } else if self.unverified_signatures.contains_voter(&signer) {
                    warn!(
                        "Duplicate unverified signatures received from {} on signed batch info",
                        signer
                    );
                    self.unverified_signatures.remove_signature(signer);
                    self.unverified_signatures
                        .add_signature(signer, signed_batch_info.signature().clone());
                } else {
                    self.unverified_signatures
                        .add_signature(signer, signed_batch_info.signature().clone());
                }
                if signer == self.info.author() {
                    self.self_voted = true;
                }
            },
            None => {
                error!(
                    "Received signature from author not in validator set: {}",
                    signed_batch_info.signer()
                );
                return Err(SignedBatchInfoError::InvalidAuthor);
            },
        }

        Ok(())
    }

    fn ready(&self, epoch_state: Arc<EpochState>) -> bool {
        let all_voters = self.all_voters();
        epoch_state
            .verifier
            .check_voting_power(all_voters.iter(), true)
            .is_ok()
    }

    fn aggregate_and_verify(
        &mut self,
        epoch_state: Arc<EpochState>,
    ) -> Result<ProofOfStore, SignedBatchInfoError> {
        if !self.ready(epoch_state.clone()) {
            return Err(SignedBatchInfoError::LowVotingPower);
        }
        if self.completed {
            panic!("Cannot call take twice, unexpected issue occurred");
        }

        let mut all_signatures = self.verified_signatures.clone();
        for (author, signature) in self.unverified_signatures.signatures() {
            all_signatures.add_signature(*author, signature.clone());
        }

        let aggregated_sig = epoch_state
            .verifier
            .aggregate_signatures(&all_signatures)
            .map_err(|e| {
                error!(
                    "Unable to aggregate signatures in proof coordinator. err = {:?}",
                    e
                );
                SignedBatchInfoError::UnableToAggregate
            })?;

        let verified_aggregate_signature = match epoch_state
            .verifier
            .verify_multi_signatures(&self.info, &aggregated_sig)
        {
            Ok(_) => aggregated_sig,
            Err(_) => {
                // Question: Should we assign min tasks per thread here for into_par_iter()?
                let verified = self
                    .unverified_signatures
                    .signatures()
                    .into_par_iter()
                    .flat_map(|(account_address, signature)| {
                        if epoch_state
                            .verifier
                            .verify(*account_address, &self.info, signature)
                            .is_ok()
                        {
                            return Some((*account_address, signature.clone()));
                        }
                        None
                    })
                    .collect::<Vec<_>>();
                for (account_address, signature) in verified {
                    self.verified_signatures
                        .add_signature(account_address, signature.clone());
                    self.unverified_signatures.remove_signature(account_address);
                }
                epoch_state.verifier.add_malicious_authors(
                    self.unverified_signatures
                        .signatures()
                        .keys()
                        .cloned()
                        .collect(),
                );
                self.unverified_signatures = PartialSignatures::empty();
                let aggregated_sig = epoch_state
                    .verifier
                    .aggregate_signatures(&self.verified_signatures)
                    .map_err(|e| {
                        error!(
                            "Unable to aggregate signatures in proof coordinator err = {:?}",
                            e
                        );
                        SignedBatchInfoError::UnableToAggregate
                    })?;
                epoch_state
                    .verifier
                    .verify_multi_signatures(&self.info, &aggregated_sig)
                    .map_err(|e| {
                        error!(
                            "Unable to verify aggregated signature in proof coordinator err = {:?}",
                            e
                        );
                        SignedBatchInfoError::InvalidAggregatedSignature
                    })?;
                aggregated_sig
            },
        };
        if self.ready(epoch_state) {
            self.completed = true;
            Ok(ProofOfStore::new(
                self.info.clone(),
                verified_aggregate_signature,
            ))
        } else {
            Err(SignedBatchInfoError::LowVotingPower)
        }
    }

    fn batch_info(&self) -> &BatchInfo {
        &self.info
    }
}

pub(crate) struct ProofCoordinator {
    peer_id: PeerId,
    proof_timeout_ms: usize,
    batch_info_to_proof: HashMap<BatchInfo, IncrementalProofState>,
    // to record the batch creation time
    batch_info_to_time: HashMap<BatchInfo, Instant>,
    timeouts: Timeouts<BatchInfo>,
    batch_reader: Arc<dyn BatchReader>,
    batch_generator_cmd_tx: tokio::sync::mpsc::Sender<BatchGeneratorCommand>,
    proof_cache: ProofCache,
    broadcast_proofs: bool,
}

//PoQS builder object - gather signed digest to form PoQS
impl ProofCoordinator {
    pub fn new(
        proof_timeout_ms: usize,
        peer_id: PeerId,
        batch_reader: Arc<dyn BatchReader>,
        batch_generator_cmd_tx: tokio::sync::mpsc::Sender<BatchGeneratorCommand>,
        proof_cache: ProofCache,
        broadcast_proofs: bool,
    ) -> Self {
        Self {
            peer_id,
            proof_timeout_ms,
            batch_info_to_proof: HashMap::new(),
            batch_info_to_time: HashMap::new(),
            timeouts: Timeouts::new(),
            batch_reader,
            batch_generator_cmd_tx,
            proof_cache,
            broadcast_proofs,
        }
    }

    fn init_proof(
        &mut self,
        signed_batch_info: &SignedBatchInfo,
    ) -> Result<(), SignedBatchInfoError> {
        // Check if the signed digest corresponding to our batch
        if signed_batch_info.author() != self.peer_id {
            return Err(SignedBatchInfoError::WrongAuthor);
        }
        let batch_author = self
            .batch_reader
            .exists(signed_batch_info.digest())
            .ok_or(SignedBatchInfoError::NotFound)?;
        if batch_author != signed_batch_info.author() {
            return Err(SignedBatchInfoError::WrongAuthor);
        }

        self.timeouts.add(
            signed_batch_info.batch_info().clone(),
            self.proof_timeout_ms,
        );
        self.batch_info_to_proof.insert(
            signed_batch_info.batch_info().clone(),
            IncrementalProofState::new(signed_batch_info.batch_info().clone()),
        );
        self.batch_info_to_time
            .entry(signed_batch_info.batch_info().clone())
            .or_insert(Instant::now());
        debug!(
            LogSchema::new(LogEvent::ProofOfStoreInit),
            digest = signed_batch_info.digest(),
            batch_id = signed_batch_info.batch_id().id,
        );
        Ok(())
    }

    fn add_signature(
        &mut self,
        signed_batch_info: SignedBatchInfo,
        epoch_state: Arc<EpochState>,
        verified: bool,
    ) -> Result<Option<ProofOfStore>, SignedBatchInfoError> {
        if !self
            .batch_info_to_proof
            .contains_key(signed_batch_info.batch_info())
        {
            self.init_proof(&signed_batch_info)?;
        }
        if let Some(value) = self
            .batch_info_to_proof
            .get_mut(signed_batch_info.batch_info())
        {
            value.add_signature(&signed_batch_info, epoch_state.clone(), verified)?;
            if !value.completed && value.ready(epoch_state.clone()) {
                let proof = {
                    let _timer = counters::SIGNED_BATCH_INFO_VERIFY_DURATION.start_timer();
                    value.aggregate_and_verify(epoch_state)?
                };
                // proof validated locally, so adding to cache
                self.proof_cache
                    .insert(proof.info().clone(), proof.multi_signature().clone());
                // quorum store measurements
                let duration = self
                    .batch_info_to_time
                    .remove(signed_batch_info.batch_info())
                    .ok_or(
                        // Batch created without recording the time!
                        SignedBatchInfoError::NoTimeStamps,
                    )?
                    .elapsed();
                counters::BATCH_TO_POS_DURATION.observe_duration(duration);
                return Ok(Some(proof));
            }
        } else {
            return Err(SignedBatchInfoError::NotFound);
        }
        Ok(None)
    }

    fn update_counters_on_expire(state: &IncrementalProofState, epoch_state: Arc<EpochState>) {
        // Count late votes separately
        if !state.completed && !state.self_voted {
            counters::BATCH_RECEIVED_LATE_REPLIES_COUNT.inc_by(state.voter_count());
            return;
        }

        counters::BATCH_RECEIVED_REPLIES_COUNT.observe(state.voter_count() as f64);
        counters::BATCH_RECEIVED_REPLIES_VOTING_POWER
            .observe(state.aggregate_voting_power(&epoch_state.verifier) as f64);
        if !state.completed {
            counters::BATCH_SUCCESSFUL_CREATION.observe(0.0);
        }
    }

    async fn expire(&mut self, epoch_state: Arc<EpochState>) {
        let mut batch_ids = vec![];
        for signed_batch_info_info in self.timeouts.expire() {
            if let Some(state) = self.batch_info_to_proof.remove(&signed_batch_info_info) {
                if !state.completed {
                    batch_ids.push(signed_batch_info_info.batch_id());
                }

                // We skip metrics if the proof did not complete and did not get a self vote, as it
                // is considered a proof that was re-inited due to a very late vote.
                if !state.completed && !state.self_voted {
                    continue;
                }

                if !state.completed {
                    counters::TIMEOUT_BATCHES_COUNT.inc();
                    info!(
                        LogSchema::new(LogEvent::IncrementalProofExpired),
                        digest = signed_batch_info_info.digest(),
                        self_voted = state.self_voted,
                    );
                }
                Self::update_counters_on_expire(&state, epoch_state.clone());
            }
        }
        if self
            .batch_generator_cmd_tx
            .send(BatchGeneratorCommand::ProofExpiration(batch_ids))
            .await
            .is_err()
        {
            warn!("Failed to send proof expiration to batch generator");
        }
    }

    pub async fn start(
        mut self,
        mut rx: Receiver<ProofCoordinatorCommand>,
        mut network_sender: impl QuorumStoreSender,
        epoch_state: Arc<EpochState>,
    ) {
        let mut interval = time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                Some(command) = rx.recv() => monitor!("proof_coordinator_handle_command", {
                    match command {
                        ProofCoordinatorCommand::Shutdown(ack_tx) => {
                            counters::QUORUM_STORE_MSG_COUNT.with_label_values(&["ProofCoordinator::shutdown"]).inc();
                            ack_tx
                                .send(())
                                .expect("Failed to send shutdown ack to QuorumStore");
                            break;
                        },
                        ProofCoordinatorCommand::CommitNotification(batches) => {
                            counters::QUORUM_STORE_MSG_COUNT.with_label_values(&["ProofCoordinator::commit_notification"]).inc();
                            for batch in batches {
                                let digest = batch.digest();
                                if let Entry::Occupied(existing_proof) = self.batch_info_to_proof.entry(batch.clone()) {
                                    if batch == *existing_proof.get().batch_info() {
                                        let incremental_proof = existing_proof.get();
                                        if incremental_proof.completed {
                                            counters::BATCH_SUCCESSFUL_CREATION.observe(1.0);
                                        } else {
                                            info!("QS: received commit notification for batch that did not complete: {}, self_voted: {}", digest, incremental_proof.self_voted);
                                        }
                                        debug!(
                                            LogSchema::new(LogEvent::ProofOfStoreCommit),
                                            digest = digest,
                                            batch_id = batch.batch_id().id,
                                            proof_completed = incremental_proof.completed,
                                        );
                                    }
                                }
                            }
                        },
                        ProofCoordinatorCommand::AppendSignature((signed_batch_infos, verified)) => {
                            let mut proofs = vec![];
                            for signed_batch_info in signed_batch_infos.take().into_iter() {
                                let peer_id = signed_batch_info.signer();
                                let digest = *signed_batch_info.digest();
                                let batch_id = signed_batch_info.batch_id();
                                match self.add_signature(signed_batch_info, epoch_state.clone(), verified) {
                                    Ok(result) => {
                                        if let Some(proof) = result {
                                            debug!(
                                                LogSchema::new(LogEvent::ProofOfStoreReady),
                                                digest = digest,
                                                batch_id = batch_id.id,
                                            );
                                            proofs.push(proof);
                                        }
                                    },
                                    Err(e) => {
                                        // Can happen if we already garbage collected, the commit notification is late, or the peer is misbehaving.
                                        if peer_id == self.peer_id {
                                            info!("QS: could not add signature from self, digest = {}, batch_id = {}, err = {:?}", digest, batch_id, e);
                                        } else {
                                            debug!("QS: could not add signature from peer {}, digest = {}, batch_id = {}, err = {:?}", peer_id, digest, batch_id, e);
                                        }
                                    },
                                }
                            }
                            if !proofs.is_empty() {
                                if self.broadcast_proofs {
                                    network_sender.broadcast_proof_of_store_msg(proofs).await;
                                } else {
                                    network_sender.send_proof_of_store_msg_to_self(proofs).await;
                                }
                            }
                        },
                    }
                }),
                _ = interval.tick() => {
                    monitor!("proof_coordinator_handle_tick", self.expire(epoch_state.clone()).await);
                }
            }
        }
    }
}
