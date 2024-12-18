use anyhow::Result;
use rust_bert::pipelines::question_answering::{
    Answer, QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use std::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::{self, JoinHandle};

/// Message type for internal channel, passing around texts and return value
/// senders
type Message = (String, String, oneshot::Sender<Vec<Vec<Answer>>>);

/// Runner for sentiment classification
#[derive(Debug, Clone)]
pub struct QuestionAnsweringClassifier {
    sender: mpsc::SyncSender<Message>,
}

impl QuestionAnsweringClassifier {
    /// Spawn a classifier on a separate thread and return a classifier instance
    /// to interact with it
    pub fn spawn() -> (JoinHandle<Result<()>>, QuestionAnsweringClassifier) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = task::spawn_blocking(move || Self::runner(receiver));
        (handle, QuestionAnsweringClassifier { sender })
    }

    /// The classification runner itself
    fn runner(receiver: mpsc::Receiver<Message>) -> Result<()> {
        // Needs to be in sync runtime, async doesn't work
        let model = QuestionAnsweringModel::new(QuestionAnsweringConfig::default())?;

        while let Ok((question, context, sender)) = receiver.recv() {
            let answers = model.predict(&[QaInput { question, context }], 1, 32);
            sender.send(answers).expect("sending results");
        }

        Ok(())
    }

    /// Make the runner predict a sample and return the result
    pub async fn predict(&self, question: String, context: String) -> Result<Vec<Vec<Answer>>> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send((question, context, sender))?;
        Ok(receiver.await?)
    }
}
