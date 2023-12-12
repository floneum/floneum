use kalosm::*;

#[tokio::main]
async fn main() {
    let mut test_cases = TestCases::new()
        .with_case("Hello", "Hello")
        .with_case("Hello", "Goodbye")
        .with_case("Hello", "1")
        .with_case("Hello", "12")
        .with_case("Hello", "123")
        .with_case("Hello", "134")
        .with_case("Hello", "14")
        .with_case("Hello", "Hello");
    let mut bert_distance = BertDistance::default();
    let distance = test_cases.evaluate(&mut bert_distance).await.normalized();
    println!("{}", distance);
}
