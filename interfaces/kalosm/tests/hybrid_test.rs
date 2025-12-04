use kalosm::language::*;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use surrealdb::{
    engine::{any::Any, local::Mem},
    Surreal,
};

/// Blog post document - represents a real-world use case
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BlogPost {
    title: String,
    body: String,
    author: String,
    tags: Vec<String>,
    published_at: String,
}

impl Hash for BlogPost {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.title.hash(state);
        self.body.hash(state);
    }
}

impl AsRef<Document> for BlogPost {
    fn as_ref(&self) -> &Document {
        // SAFETY: This is a workaround for testing purposes.
        static mut DOC: Option<Document> = None;
        unsafe {
            DOC = Some(Document::from_parts(&self.title, &self.body));
            DOC.as_ref().unwrap()
        }
    }
}

/// Product document - e-commerce use case
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Product {
    name: String,
    description: String,
    category: String,
    price: f64,
    sku: String,
}

impl Hash for Product {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sku.hash(state);
    }
}

impl AsRef<Document> for Product {
    fn as_ref(&self) -> &Document {
        static mut DOC: Option<Document> = None;
        unsafe {
            DOC = Some(Document::from_parts(&self.name, &self.description));
            DOC.as_ref().unwrap()
        }
    }
}

/// Knowledge base article - documentation/FAQ use case
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct KnowledgeArticle {
    question: String,
    answer: String,
    category: String,
    helpful_count: i32,
}

impl Hash for KnowledgeArticle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.question.hash(state);
    }
}

impl AsRef<Document> for KnowledgeArticle {
    fn as_ref(&self) -> &Document {
        static mut DOC: Option<Document> = None;
        unsafe {
            let combined = format!("{} {}", self.question, self.answer);
            DOC = Some(Document::from_parts(&self.question, combined));
            DOC.as_ref().unwrap()
        }
    }
}

// Helper to create test database
async fn setup_db() -> Result<Surreal<Any>, Box<dyn std::error::Error>> {
    let db = surrealdb::engine::any::connect("mem://").await?;
    db.use_ns("test").use_db("test").await?;

    Ok(db)
}

// Helpers to create sample documents
fn sample_blog_posts() -> Vec<BlogPost> {
    vec![
    BlogPost {
        title: "Getting Started with Rust Async".to_string(),
        body: "Async programming in Rust uses the async/await syntax. This allows you to write asynchronous code that looks like synchronous code. Tokio is the most popular async runtime for Rust.".to_string(),
        author: "Alice".to_string(),
        tags: vec!["rust".to_string(), "async".to_string(), "tutorial".to_string()],
        published_at: "2024-01-15".to_string(),
    },
    BlogPost {
        title: "Understanding Rust Ownership".to_string(),
        body: "Ownership is Rust's most unique feature. It enables Rust to make memory safety guarantees without needing a garbage collector. Every value has a variable that's its owner.".to_string(),
        author: "Bob".to_string(),
        tags: vec!["rust".to_string(), "basics".to_string()],
        published_at: "2024-01-10".to_string(),
    },
    BlogPost {
        title: "Building REST APIs with Python Flask".to_string(),
        body: "Flask is a lightweight WSGI web application framework in Python. It's designed to make getting started quick and easy, with the ability to scale up to complex applications.".to_string(),
        author: "Charlie".to_string(),
        tags: vec!["python".to_string(), "web".to_string(), "api".to_string()],
        published_at: "2024-01-20".to_string(),
    },
    BlogPost {
        title: "Introduction to Vector Databases".to_string(),
        body: "Vector databases store data as high-dimensional vectors, which are mathematical representations of features. They enable similarity search and are crucial for AI applications like RAG.".to_string(),
        author: "Alice".to_string(),
        tags: vec!["database".to_string(), "ai".to_string(), "vectors".to_string()],
        published_at: "2024-02-01".to_string(),
    },
    BlogPost {
        title: "Python vs Rust: Performance Comparison".to_string(),
        body: "While Python is easier to learn and faster to write, Rust offers superior performance and memory safety. The choice depends on your use case and team expertise.".to_string(),
        author: "Bob".to_string(),
        tags: vec!["rust".to_string(), "python".to_string(), "comparison".to_string()],
        published_at: "2024-02-05".to_string(),
    },
]
}

fn sample_products() -> Vec<Product> {
    vec![
    Product {
        name: "Mechanical Keyboard".to_string(),
        description: "Professional mechanical keyboard with Cherry MX Blue switches. Perfect for programmers who love tactile feedback. RGB backlit with customizable keys.".to_string(),
        category: "Electronics".to_string(),
        price: 129.99,
        sku: "KB-001".to_string(),
    },
    Product {
        name: "Ergonomic Office Chair".to_string(),
        description: "Premium ergonomic office chair with lumbar support. Adjustable height and armrests. Mesh back for breathability during long coding sessions.".to_string(),
        category: "Furniture".to_string(),
        price: 349.99,
        sku: "CH-001".to_string(),
    },
    Product {
        name: "USB-C Docking Station".to_string(),
        description: "Universal USB-C dock with dual 4K monitor support. Includes multiple USB ports, ethernet, and power delivery. Compatible with MacBook and laptops.".to_string(),
        category: "Electronics".to_string(),
        price: 199.99,
        sku: "DK-001".to_string(),
    },
    Product {
        name: "Standing Desk Converter".to_string(),
        description: "Height adjustable standing desk converter. Easy to use, fits on existing desk. Promotes better posture and reduces back pain from sitting.".to_string(),
        category: "Furniture".to_string(),
        price: 249.99,
        sku: "SD-001".to_string(),
    },
    Product {
        name: "Wireless Mouse".to_string(),
        description: "Ergonomic wireless mouse with precision tracking. Long battery life and comfortable grip for extended use. Works on any surface.".to_string(),
        category: "Electronics".to_string(),
        price: 49.99,
        sku: "MS-001".to_string(),
    },
]
}

fn sample_kb_articles() -> Vec<KnowledgeArticle> {
    vec![
    KnowledgeArticle {
        question: "How do I reset my password?".to_string(),
        answer: "To reset your password, click the 'Forgot Password' link on the login page. Enter your email address and we'll send you a reset link. The link expires in 24 hours.".to_string(),
        category: "Account".to_string(),
        helpful_count: 45,
    },
    KnowledgeArticle {
        question: "What payment methods do you accept?".to_string(),
        answer: "We accept Visa, MasterCard, American Express, PayPal, and bank transfers. All payments are processed securely through our payment gateway.".to_string(),
        category: "Billing".to_string(),
        helpful_count: 32,
    },
    KnowledgeArticle {
        question: "How long does shipping take?".to_string(),
        answer: "Standard shipping takes 5-7 business days. Express shipping is 2-3 business days. International orders may take 10-14 days depending on customs.".to_string(),
        category: "Shipping".to_string(),
        helpful_count: 28,
    },
    KnowledgeArticle {
        question: "Can I change my order after placing it?".to_string(),
        answer: "You can modify your order within 1 hour of placement. After that, the order enters processing and cannot be changed. Contact support for assistance.".to_string(),
        category: "Orders".to_string(),
        helpful_count: 19,
    },
    KnowledgeArticle {
        question: "What is your return policy?".to_string(),
        answer: "We offer 30-day returns on most items. Products must be unused and in original packaging. Refunds are processed within 5-7 business days after we receive the return.".to_string(),
        category: "Returns".to_string(),
        helpful_count: 56,
    },
]
}

#[tokio::test]
async fn test_blog_semantic_search() -> Result<(), Box<dyn std::error::Error>> {
    let db = setup_db().await?;

    // Build hybrid table for blog posts
    let table = db
        .document_table_builder("blog_posts")
        .with_hybrid_search("body")
        .build::<BlogPost>()
        .await?;

    // Insert sample posts
    for post in sample_blog_posts() {
        table.insert(post).await?;
    }

    // Search for async programming content
    let results = table
        .hybrid_search("asynchronous programming tutorials")
        .with_results(3)
        .run_weighted()
        .await?;

    assert!(!results.is_empty(), "Should find async-related posts");

    println!("\n=== Async Programming Search ===");
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. {} (score: {:.3}, sem: {:.3}, kw: {:.3})",
            i + 1,
            result.record.title,
            result.score,
            result.semantic_score,
            result.keyword_score
        );
    }

    // Check that at least one of the top results is about async/Rust
    let has_relevant = results.iter().take(2).any(|r| {
        r.record.title.to_lowercase().contains("async")
            || r.record.title.to_lowercase().contains("rust")
            || r.record.body.to_lowercase().contains("async")
    });

    assert!(
        has_relevant,
        "At least one of the top 2 results should be about async/Rust programming. Got: {:?}",
        results
            .iter()
            .take(2)
            .map(|r| &r.record.title)
            .collect::<Vec<_>>()
    );

    Ok(())
}
