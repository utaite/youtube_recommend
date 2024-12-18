use crate::youtube_transcript::error;
use crate::youtube_transcript::utils::to_human_readable;
use roxmltree::Document;
use serde;
use serde::Deserialize;
use serde::Serialize;
use serde_json;
use std::error::Error;
use std::time::Duration;

#[derive(Default, Deserialize)]
pub(crate) struct Caption {
    #[serde(rename(deserialize = "baseUrl"))]
    pub base_url: String,
    #[serde(rename(deserialize = "languageCode"))]
    pub lang_code: String,
}

#[derive(Deserialize)]
struct Captions {
    #[serde(rename(deserialize = "captionTracks"))]
    caption_tracks: Vec<Caption>,
}

pub(crate) trait HTMLParser<'a> {
    fn html_string(&'a self) -> &'a str;

    fn caption(&'a self, from: &str, to: &str) -> Result<Caption, error::Error> {
        let html = self.html_string();
        let start = html
            .split_once(from)
            .ok_or_else(|| error::Error::ParseError(format!("Cannot parse html for: {}", from)))?
            .1;
        let actual_json = start
            .split_once(to)
            .ok_or_else(|| error::Error::ParseError(format!("Cannot parse html to: {}", to)))?
            .0;
        let value: Captions = serde_json::from_str(actual_json)
            .map_err(|x| error::Error::ParseError(format!("{}", x)))?;
        let caption = value
            .caption_tracks
            .into_iter()
            .filter(|x| x.lang_code == "ko")
            .next()
            .unwrap_or_default();
        Ok(caption)
    }
}

impl<'a> HTMLParser<'a> for String {
    fn html_string(&'a self) -> &'a str {
        self.as_str()
    }
}

impl<'a> HTMLParser<'a> for str {
    fn html_string(&'a self) -> &'a str {
        self
    }
}

/// Struct that contains data about transcirpt text along with start and duration in the whole video.
#[derive(PartialEq, Debug, Serialize)]
pub struct TranscriptCore {
    /// transcript text. Ex: "Hi How are you"
    pub text: String,
    /// starting time of the text in the whole video. Ex: "0 sec"
    pub start: Duration,
    /// duration of the text Ex: "0.8 sec"
    pub duration: Duration,
}

/// Struct containing youtube's transcript data as a Vec<[`TranscriptCore`]>
#[derive(Serialize)]
pub struct Transcript {
    /// List of transcript texts in [`TranscriptCore`] format
    pub transcripts: Vec<TranscriptCore>,
}

impl IntoIterator for Transcript {
    type IntoIter = <Vec<TranscriptCore> as IntoIterator>::IntoIter;
    type Item = TranscriptCore;

    fn into_iter(self) -> Self::IntoIter {
        self.transcripts.into_iter()
    }
}

impl From<Transcript> for String {
    fn from(value: Transcript) -> Self {
        {
            value
                .transcripts
                .into_iter()
                .map(|x| {
                    let start_h = to_human_readable(&x.start);
                    let dur_h = to_human_readable(&x.duration);
                    format!(
                        "\nstart at: {} for duration {}\n{}\n==========\n\n",
                        start_h, dur_h, x.text
                    )
                })
                .collect::<String>()
        }
    }
}

pub(crate) struct TranscriptParser;

impl TranscriptParser {
    pub fn parse<'input>(
        transcript: &'input Document<'input>,
    ) -> Result<Transcript, Box<dyn Error>> {
        let mut transcripts = Vec::new();
        let nodes = transcript
            .descendants()
            .filter(|x| x.tag_name() == "text".into());
        for node in nodes {
            let start = node
                .attribute("start")
                .ok_or(error::Error::ParseError("transcript parse error".into()))?
                .parse::<f32>()?;
            let duration = node
                .attribute("dur")
                .ok_or(error::Error::ParseError("transcript parse error".into()))?
                .parse::<f32>()?;
            let node = node
                .last_child()
                .ok_or(error::Error::ParseError("transcript parse error".into()))?;
            let text = node
                .text()
                .ok_or(error::Error::ParseError("transcript error".into()))?;

            transcripts.push(TranscriptCore {
                text: text.into(),
                start: Duration::from_secs_f32(start),
                duration: Duration::from_secs_f32(duration),
            })
        }
        Ok(Transcript { transcripts })
    }
}
