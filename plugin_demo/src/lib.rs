use core::panic;
use std::vec;

use crate::exports::plugins::main::definitions::*;
use crate::plugins::main::imports::*;

wit_bindgen::generate!(in "../wit");

struct Plugin;

export_plugin_world!(Plugin);

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "test plugin".to_string(),
            description: "this is a test plugin".to_string(),
            inputs: vec![IoDefinition {
                name: "input".to_string(),
                ty: ValueType::Text,
            }],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Text,
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let models = [
            ModelType::Llama(LlamaType::Vicuna),
            ModelType::Llama(LlamaType::Guanaco),
            // doesn't work with embeddings
            ModelType::GptNeoX(GptNeoXType::DollySevenB),
            ModelType::GptNeoX(GptNeoXType::TinyPythia),
            ModelType::GptNeoX(GptNeoXType::LargePythia),
            ModelType::Mpt(MptType::Chat),
            ModelType::Mpt(MptType::Story),
            ModelType::Mpt(MptType::Instruct),
            ModelType::Mpt(MptType::Base),
        ];

        {
            let session = load_model(ModelType::Llama(LlamaType::Vicuna));

            let texts: Vec<_> = ARTICLE.split(".").collect();
            let embeddings: Vec<_> = texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    print(&format!("{}/{}:{text}\n", i, texts.len()));
                    get_embedding(session, text)
                })
                .collect();
            let borrowed_embeddings: Vec<_> = embeddings.iter().collect();
            let db: EmbeddingDbId = create_embedding_db(&borrowed_embeddings, &texts);
            let _results = find_closest_documents(db, &embeddings[0], 10);
            unload_model(session);
        }

        let mut outputs = String::new();

        for model in models {
            let session = load_model(model);

            let text_input = match &input[0] {
                Value::Text(text) => text,
                _ => panic!("expected text input"),
            };

            let text_input = format!("This is a chat between an AI chatbot and a human. The chatbot is programmed to be extremly helpful and always attempt to answer correctly. The human will start questions with ### Human; the AI with start answers with ### Assistant\n### Human{text_input}\n ### Assistant");

            let responce = infer(session, &text_input, Some(50), Some("### Human"));

            let text = format!("{model:?}:\n{}\n\n", responce);
            print(&text);
            outputs += &text;

            unload_model(session);
        }

        vec![Value::Text(outputs)]
    }
}

const ARTICLE: &str = r##"Leaders of both parties set up a final vote on Thursday night to avoid a default, after putting down a revolt by some senators who raised concerns that the debt-limit package would under-fund the Pentagon.

    Give this article
    
    
    557
    Senator Chuck Schumer emerging from a black car holding a coffee cup and wearing a dark suit and a pink tie.
    Senator Chuck Schumer, Democrat of New York and the majority leader, said the Senate would remain in session until it approved the package.Credit...Kenny Holston/The New York Times
    
    Carl Hulse
    By Carl Hulse
    Reporting from Capitol Hill
    
    June 1, 2023
    Updated 7:41 p.m. ET
    The Senate was on track on Thursday night to pass bipartisan legislation suspending the debt limit and imposing new spending caps and send it to President Biden, racing to avoid a default.
    
    After an afternoon of closed-door talks to resolve a last-minute dispute over Pentagon funding, Senator Chuck Schumer, Democrat of New York and the majority leader, said the chamber would begin voting on proposed changes to the package — all of which were expected to fail — and pass it later Thursday night, days before the Treasury Department is set to run out of cash to pay its bills.
    
    “America can breathe a sigh of relief because we are avoiding default,” Mr. Schumer said.
    
    The announcement came after a day of uncertainty as a handful of Republicans complained that the deal — negotiated between Mr. Biden and Speaker Kevin McCarthy without input from the Senate — would under-fund the military, and demanded a commitment that their concerns would be addressed before it could be passed.
    
    By evening, Senate officials and Senator Lindsey Graham, the South Carolina Republican who had been a chief critic of the Pentagon spending levels, said leaders in both parties had negotiated language that reassured him and other defense hawks sufficiently to back the bill, clearing the way for final votes.
    
    “It does not fix this bill totally, but it is a march in the right direction,” said Mr. Graham.
    
    The debt-limit agreement, which was approved overwhelmingly by the House on Wednesday night, would suspend the $31.4 trillion debt ceiling until January 2025 while cutting spending on domestic programs.
    
    It would increase Pentagon spending to $886 billion for next year, a 3 percent raise, but G.O.P. backers of higher spending for the military noted that that would not keep pace with inflation, and argued that the package fell far short of what was needed.
    
    Understand the Debt Limit Deal
    Card 1 of 5
    Lifting the debt ceiling. The deal reached by President Biden and Speaker Kevin McCarthy would suspend the nation’s debt limit until January 2025. This would allow the government to keep borrowing money so it can pay its bills on time.
    
    Spending caps and cuts. In exchange for suspending the debt ceiling, Republicans demanded a range of concessions. Chief among them are caps on some spending over the next two years. The deal also claws back $10 billion in I.R.S. funding.
    
    Food stamps. The bill would place additional work requirements on older Americans who receive assistance through the Supplemental Nutrition Assistance Program, but it also would expand food stamp access for veterans and homeless people.
    
    Student loans. The legislation would officially end Biden’s freeze on student loan repayments by the end of summer. It would also prevent the president from issuing another last-minute extension, as he has done several times.
    
    Environmental impact. Both sides agreed to new measures to get energy projects approved more quickly. The deal includes a win for Senator Joe Manchin of West Virginia, a Democrat who strongly supports fossil fuels, by fast-tracking the construction of a contentious pipeline.
    
    “To my House colleagues, I can’t believe you did this,” Mr. Graham said earlier in the day, accusing the architects of the measure of undercutting the military at a time of rising threats from Russia and China. “This budget is a win for China.”
    
    Mr. Graham and others insisted said that at a minimum, they wanted a commitment that Congress would later move on an additional funding bill to beef up the spending, although this would in effect reduce the savings Republicans had hoped to achieve through their debt limit deal.
    
    “We know that this budget is not adequate to the global threats that we face,” said Senator Susan Collins of Maine, the senior Republican on the full Appropriations Committee. “An emergency supplemental must be coming our way.”
    
    The opposition erupted almost immediately after Mr. Schumer opened the Senate on Thursday morning by warning that the chamber needed to move quickly and make no changes to the agreement to clear it for Mr. Biden’s signature by Monday. He admonished lawmakers not to engage in brinkmanship before the so-called X-date of June 5, when Treasury Secretary Janet L. Yellen has said the government will default without action by Congress.
    
    “Time is a luxury the Senate does not have if we want to prevent default,” Mr. Schumer said. “June 5 is less than four days away. At this point, any needless delay or any last-minute holdups would be an unnecessary and even dangerous risk.”
    
    Even as the deal migrated across the Capitol, the effects of the debt limit continued to pinch. The Treasury announced on Thursday that it would delay auctions of three-month and six-month “bills” — short-term debt that the government no longer has room to take on until the borrowing cap is suspended.
    
    Image
    Senator Mitch McConnell in front of reporters.
    Senator Mitch McConnell urged his fellow Republicans to back the plan.Credit...Haiyun Jiang for The New York Times
    
    As part of the deal to move forward with final votes on the bill, multiple senators secured votes on proposed changes. Mr. Schumer was determined to defeat all of them, as any alteration would force the measure back to the House, where no action would be likely to occur before the default deadline.
    
    “Any change to this bill that forces us to send it back to the House would be entirely unacceptable,” he said. “It would almost guarantee default.”
    
    The Fight Over the Debt Limit
    Biden’s Strategy: With the House passing legislation to suspend the debt ceiling and set spending limits, here’s a look at how President Biden both won and lost the negotiation.
    Defying Expectations: Speaker Kevin McCarthy delivered a debt limit agreement that few thought he could manage, but left some of his Republican colleagues feeling betrayed.
    Next Up: The bill moved to the Senate, where leaders of both parties were racing to shut down efforts to derail it before a potential default.
    A Larger Issue: The standoff has raised questions about whether there is a way to preclude such situations from happening again — by abolishing the debt ceiling or using the 14th Amendment to render the statutory limit unconstitutional.
    Among those securing a vote was Senator Tim Kaine, Democrat of Virginia, who on Thursday called for stripping a provision from the legislation that would expedite the approval of an oil pipeline in West Virginia.
    
    “I support improving the permitting process for all energy projects,” Mr. Kaine said. “But Congress putting its thumb on the scale so that one specific project doesn’t have to comply with the same process as everyone else is the definition of unfair and opens the door to corruption.”
    
    After driving much of the legislative agenda the previous two years, the Senate left negotiating on the debt limit to Mr. Biden and Mr. McCarthy, whose demand for spending cuts and other policy changes brought the country to the brink of default. Nearly all Republican senators signed a letter backing Mr. McCarthy in the effort. As a result, senators had little influence over the negotiations and are now being forced to approve legislation they did not help shape. It is leaving some frustrated.
    
    Senator John Cornyn, Republican of Texas, praised Mr. McCarthy’s efforts but said senators had no obligation to simply rubber stamp the deal and deserved opportunities to change it.
    
    “We weren’t a party to the agreement,” he said. “Why should we be bound by the strict terms of that agreement? The Senate has not had a say in the process so far.”
    
    But Senator Mitch McConnell of Kentucky, the minority leader, urged his fellow Republicans to back the plan.
    
    “Last night, an overwhelming majority of our House colleagues voted to pass the agreement Speaker McCarthy reached with President Biden,” he said. “In doing so, they took an urgent and important step in the right direction for the health of our economy and the future of our country.”"##;
