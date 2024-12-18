use crate::youtube_transcript::config::{Config, CONFIG_VAL};
use crate::youtube_transcript::parser::{HTMLParser, Transcript, TranscriptParser};
use reqwest::Client;
use roxmltree::Document;
use std::error::Error;

/// Youtube container that holds the [`Config`].
pub struct Youtube<'b> {
    config: &'b Config,
}

impl<'b> Youtube<'b> {
    /// extracts [`Transcript`] from the video link provided.
    pub async fn transcript<'a>(&self, url: &'a str) -> Result<Transcript, Box<dyn Error>> {
        let client = Client::default();
        let response = client.get(url).send().await?;
        let text = response.text().await?;
        self.transcript_from_text(&text).await
    }
    /// extracts [`Transcript`] from the youtube raw html text provided.
    pub async fn transcript_from_text(&self, text: &str) -> Result<Transcript, Box<dyn Error>> {
        let client = Client::default();
        let c = text.caption(self.config.parser.from, self.config.parser.to)?;
        if c.base_url.is_empty() {
            Ok(Transcript {
                transcripts: Vec::new(),
            })
        } else {
            let response = client.get(c.base_url).send().await?;
            let trans_resp = response.text().await?;
            let doc = Document::parse(&trans_resp)?;
            let t = TranscriptParser::parse(&doc)?;
            Ok(t)
        }
    }
}

/// Builder struct for building [`Youtube`]
pub struct YoutubeBuilder<'b> {
    config: &'b Config,
}

impl<'b> YoutubeBuilder<'b> {
    /// creates [`YoutubeBuilder`] with default [`Config`] values.
    pub fn default() -> Self {
        Self {
            config: &CONFIG_VAL,
        }
    }

    /// Builds [`Youtube`]
    pub fn build(&'b self) -> Youtube<'b> {
        Youtube {
            config: self.config,
        }
    }
}
