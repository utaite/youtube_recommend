pub mod classifier;
#[cfg(test)]
pub mod classifier_test;
pub mod youtube_transcript;

use crate::classifier::keyword_extraction_classifier::KeywordExtractionClassifier;
use crate::classifier::question_answering_classifier::QuestionAnsweringClassifier;
use crate::classifier::sentiment_classifier::SentimentClassifier;
use crate::classifier::summarization_classifier::SummarizationClassifier;
use crate::youtube_transcript::youtube::YoutubeBuilder;
use dotenv::dotenv;
use futures::future::join_all;
use reqwest::Url;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;

/*
const MAX_RESULTS_VIDEO: usize = 50;
const MAX_RESULTS_COMMENT: usize = 100;
*/

// 최대 비디오 수
const MAX_RESULTS_VIDEO: usize = 10;
// 최대 댓글 수
const MAX_RESULTS_COMMENT: usize = 100;

#[tokio::main]
async fn main() {
    dotenv().ok();
    let input_text = get_input_text();
    let videos = get_videos(input_text).await.unwrap();
    println!("{:#?}", videos);

    let scripts = get_scripts(&videos).await.unwrap();
    let comments = get_comments(&videos).await.unwrap();

    for i in 0..MAX_RESULTS_VIDEO {
        let video_value = &videos[i];
        let comments_value = &comments[i];

        let title = video_value["snippet"]["title"]
            .clone()
            .as_str()
            .unwrap()
            .to_string();
        let script = scripts[i].clone();
        let comments: Vec<String> = comments_value
            .iter()
            .map(|x| {
                x["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    .clone()
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        println!(
            "제목: {:#?}\n스크립트: {:#?}\n댓글: {:#?}",
            title, script, comments
        );

        let translate_script =
            get_translate_text(script.clone(), "KO".to_owned(), "EN".to_owned()).await;
        let translate_comments: Vec<String> = join_all(
            comments
                .iter()
                .map(|x| get_translate_text(x.clone(), "KO".to_owned(), "EN".to_owned())),
        )
        .await
        .into_iter()
        .collect();
        println!(
            "영문 스크립트: {:#?}\n영문 댓글: {:#?}",
            translate_script, translate_comments
        );

        // 1. 질의응답
        // DistilBERT 모델을 활용하여 유튜브 스크립트의 주제와 결론에 대해 질문한다.
        let (_, question_answering_classifier) = QuestionAnsweringClassifier::spawn();
        let question = "What is the theme and conclusion of the video?".to_owned();
        let translate_answers = question_answering_classifier
            .predict(question, translate_script.clone())
            .await
            .unwrap();
        let answers = join_all(translate_answers.iter().flat_map(|x| {
            x.iter()
                .map(|y| get_translate_text(y.answer.clone(), "EN".to_owned(), "KO".to_owned()))
        }))
        .await;
        println!("영문 주제와 결론: {translate_answers:#?}");
        println!("주제와 결론: {answers:#?}");

        // 2. 요약
        // BART 모델을 활용하여 유튜브 스크립트 요약을 진행한다.
        let (_, summarization_classifier) = SummarizationClassifier::spawn();
        let translate_summarize = summarization_classifier
            .summarize(vec![translate_script.clone()])
            .await
            .unwrap();
        let summarize = join_all(
            translate_summarize
                .iter()
                .map(|x| get_translate_text(x.clone(), "EN".to_owned(), "KO".to_owned())),
        )
        .await;
        println!("영문 스크립트 요약: {translate_summarize:#?}");
        println!("스크립트 요약: {summarize:#?}");

        // 3. 감정 분석
        // DistilBERT 모델을 활용하여 유튜브 댓글에 대한 이진 감정을 분석한다.
        let (_, sentiment_classifier) = SentimentClassifier::spawn();
        let sentiments = sentiment_classifier
            .predict(translate_comments.clone())
            .await
            .unwrap();
        println!("댓글 감성 분석: {sentiments:#?}");

        // 4. 키워드 추출
        // 유튜브 스크립트와 댓글에서 키워드를 추출한다.
        let (_, keyword_extraction_classifier) = KeywordExtractionClassifier::spawn();
        let translate_script_keywords = keyword_extraction_classifier
            .predict(vec![translate_script.clone()])
            .await
            .unwrap();
        let script_keywords = join_all(translate_script_keywords.iter().flat_map(|x| {
            x.iter()
                .map(|y| get_translate_text(y.text.clone(), "EN".to_owned(), "KO".to_owned()))
        }))
        .await;
        println!("영문 스크립트 키워드: {translate_script_keywords:#?}");
        println!("스크립트 키워드: {script_keywords:#?}");

        let translate_comments_keywords = keyword_extraction_classifier
            .predict(translate_comments.clone())
            .await
            .unwrap();
        let comments_keywords = join_all(translate_comments_keywords.iter().flat_map(|x| {
            x.iter()
                .map(|y| get_translate_text(y.text.clone(), "EN".to_owned(), "KO".to_owned()))
        }))
        .await;
        println!("영문 댓글 키워드: {translate_comments_keywords:#?}");
        println!("댓글 키워드: {comments_keywords:#?}");
    }
}

fn get_input_text() -> String {
    print!("어떤 제품을 추천 받으시겠습니까?: ");
    std::io::stdout().flush().unwrap();

    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    buf.trim().to_string()
}

async fn get_videos(input_text: String) -> Option<Vec<Value>> {
    let youtube_api_key = std::env::var("YOUTUBE_API_KEY").unwrap();
    let client = reqwest::Client::new();

    let mut page_token = String::new();
    let mut videos = Vec::new();

    loop {
        let params: HashMap<&str, String> = [
            ("key", youtube_api_key.clone()),
            ("maxResults", MAX_RESULTS_VIDEO.to_string()),
            ("part", "snippet".to_owned()),
            ("pageToken", page_token.clone()),
            ("q", input_text.to_owned()),
            ("type", "video".to_owned()),
        ]
        .into();
        let url =
            Url::parse_with_params("https://www.googleapis.com/youtube/v3/search", params).unwrap();
        let response = client.get(url).send().await.unwrap();

        if !response.status().is_success() {
            println!("status: {}", response.status());
            println!("text: {}", response.text().await.unwrap());
            return Some(Vec::new());
        }

        let json = response.json::<Value>().await.unwrap();

        if let Some(error) = json.get("error") {
            println!("error: {error:#?}");
            return Some(Vec::new());
        }

        if let Some(items) = json["items"].as_array() {
            videos.extend(items.clone());
        }

        if videos.len() >= MAX_RESULTS_VIDEO {
            return Some(videos);
        } else if let Some(next_page_token) = json["nextPageToken"].as_str() {
            page_token = next_page_token.to_string();
        } else {
            return Some(videos);
        }
    }
}

async fn get_scripts(videos: &Vec<Value>) -> Option<Vec<String>> {
    let youtube_loader = YoutubeBuilder::default();
    let youtube_loader = youtube_loader.build();

    let mut scripts = Vec::new();

    for video in videos {
        let id = video["id"]["videoId"].clone();
        let transcript = youtube_loader
            .transcript(("https://www.youtube.com/watch?v=".to_owned() + id.as_str()?).as_str())
            .await
            .unwrap();

        let script = transcript
            .transcripts
            .into_iter()
            .map(|x| x.text)
            .collect::<Vec<String>>()
            .join(" ");
        scripts.push(script);
    }

    Some(scripts)
}

async fn get_comments(videos: &Vec<Value>) -> Option<Vec<Vec<Value>>> {
    let youtube_api_key = std::env::var("YOUTUBE_API_KEY").unwrap();
    let client = reqwest::Client::new();

    let mut comments = Vec::new();

    for video in videos {
        let mut page_token = String::new();
        let mut comment = Vec::new();
        let id = video["id"]["videoId"].clone();

        loop {
            let params: HashMap<&str, String> = [
                ("key", youtube_api_key.clone()),
                ("maxResults", MAX_RESULTS_COMMENT.to_string()),
                ("part", "snippet".to_owned()),
                ("pageToken", page_token.clone()),
                ("video_id", id.as_str()?.to_string()),
            ]
            .into();
            let url = Url::parse_with_params(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params,
            )
            .unwrap();
            let response = client.get(url).send().await.unwrap();

            if !response.status().is_success() {
                println!("status: {}", response.status());
                println!("text: {}", response.text().await.unwrap());
                comments.push(Vec::new());
                break;
            }

            let json = response.json::<Value>().await.unwrap();

            if let Some(error) = json.get("error") {
                println!("error: {error:#?}");
                comments.push(Vec::new());
                break;
            }

            if let Some(items) = json["items"].as_array() {
                comment.extend(items.clone());
            }

            if comment.len() >= MAX_RESULTS_COMMENT {
                comments.push(comment);
                break;
            } else if let Some(next_page_token) = json["nextPageToken"].as_str() {
                page_token = next_page_token.to_string();
            } else {
                comments.push(comment);
                break;
            }
        }
    }

    Some(comments)
}

async fn get_translate_text(text: String, source: String, target: String) -> String {
    let deepl_api_key = std::env::var("DEEPL_API_KEY").unwrap();
    let client = reqwest::Client::new();

    let body: HashMap<&str, Value> = [
        ("text", [text].into()),
        ("source_lang", source.into()),
        ("target_lang", target.into()),
    ]
    .into();

    let url = Url::parse("https://api-free.deepl.com/v2/translate").unwrap();
    let response = client
        .post(url)
        .header(
            "Authorization",
            "DeepL-Auth-Key ".to_string() + deepl_api_key.as_str(),
        )
        .json(&body)
        .send()
        .await
        .unwrap();

    if !response.status().is_success() {
        println!("status: {}", response.status());
        println!("text: {}", response.text().await.unwrap());
        return String::new();
    }

    let json = response.json::<Value>().await.unwrap();

    if let Some(translations) = json["translations"].as_array() {
        translations[0]["text"]
            .clone()
            .as_str()
            .unwrap()
            .to_string()
    } else {
        String::new()
    }
}
