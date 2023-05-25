// async fn search(query: &str) -> String {
//      let url = format!("https://www.google.com/search?q={query}");
//     let response = reqwest::get(&url).await.unwrap();
//     let raw_html=response.text().await.unwrap();

//     let fragment = Html::parse_document(&raw_html);
//     let selector = Selector::parse("a").unwrap();
//     let urls = fragment.select(&selector).filter_map(|t|t.value().attr("href").filter(|url|url.contains("http") && !url.contains("google")));
//     for url in urls{
//         println!("{}", url);
//         if let Ok(all_ps) = reqwest::get(url).await.map(|r|r.text()){
//             if let Ok(all_ps) = all_ps.await{
//                 let fragment = Html::parse_document(&all_ps);
//                 let ps = fragment.select(&Selector::parse("p").unwrap()).map(|t|t.inner_html()).collect::<Vec<_>>();
//                 let text = ps.join("\n");
//                 println!("{}", text);
//                 if text.len() > 100 {
//                     return text;
//                 }
//             }
//         }
//     }
//     String::new()
// }