use anyhow::Result;
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use std::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::{self, JoinHandle};

/// Message type for internal channel, passing around texts and return value
/// senders
type Message = (Vec<String>, oneshot::Sender<Vec<String>>);

/// Runner for sentiment classification
#[derive(Debug, Clone)]
pub struct SummarizationClassifier {
    sender: mpsc::SyncSender<Message>,
}

impl SummarizationClassifier {
    /// Spawn a classifier on a separate thread and return a classifier instance
    /// to interact with it
    pub fn spawn() -> (JoinHandle<Result<()>>, SummarizationClassifier) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = task::spawn_blocking(move || Self::runner(receiver));
        (handle, SummarizationClassifier { sender })
    }

    /// The classification runner itself
    fn runner(receiver: mpsc::Receiver<Message>) -> Result<()> {
        // Needs to be in sync runtime, async doesn't work
        let model = SummarizationModel::new(SummarizationConfig::default())?;

        while let Ok((texts, sender)) = receiver.recv() {
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            let summarize = model.summarize(&texts)?;
            sender.send(summarize).expect("sending results");
        }

        Ok(())
    }

    /// Make the runner predict a sample and return the result
    pub async fn summarize(&self, texts: Vec<String>) -> Result<Vec<String>> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send((texts, sender))?;
        Ok(receiver.await?)
    }
}
