use crate::classifier::keyword_extraction_classifier::KeywordExtractionClassifier;
use crate::classifier::question_answering_classifier::QuestionAnsweringClassifier;
use crate::classifier::sentiment_classifier::SentimentClassifier;
use crate::classifier::summarization_classifier::SummarizationClassifier;

#[tokio::test]
async fn question_answering_classifier_test() {
    let (_, question_answering_classifier) = QuestionAnsweringClassifier::spawn();
    let question = String::from("Where does Amy live?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = question_answering_classifier
        .predict(question, context)
        .await
        .unwrap();
    println!("answers: {answers:?}");
}

#[tokio::test]
async fn summarization_classifier_test() {
    let (_, summarization_classifier) = SummarizationClassifier::spawn();
    let texts = vec![
        "In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \"
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b.".to_owned(),
    ];

    let summarize = summarization_classifier.summarize(texts).await.unwrap();
    println!("summarize: {summarize:?}");
}

#[tokio::test]
async fn sentiment_classifier_test() {
    let (_, sentiment_classifier) = SentimentClassifier::spawn();
    let texts = vec![
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.".to_owned(),
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...".to_owned(),
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.".to_owned(),
    ];

    let sentiments = sentiment_classifier.predict(texts).await.unwrap();
    println!("sentiments: {sentiments:?}");
}

#[tokio::test]
async fn keyword_extraction_classifier_test() {
    let (_, keyword_extraction_classifier) = KeywordExtractionClassifier::spawn();
    let texts = vec![
        "Rust is a multi-paradigm, general-purpose programming language. \
       Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
       that all references point to valid memory—without requiring the use of a garbage collector or \
       reference counting present in other memory-safe languages. To simultaneously enforce \
       memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
       and variable scope of all references in a program during compilation. Rust is popular for \
       systems programming but also offers high-level features including functional programming constructs.".to_owned(),
    ];

    let keywords = keyword_extraction_classifier.predict(texts).await.unwrap();
    println!("keywords: {keywords:?}");
}
