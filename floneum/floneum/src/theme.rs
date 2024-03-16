use floneumite::Category;

pub fn category_bg_color(category: Category) -> &'static str {
    match category {
        Category::Utility => "bg-[#88F575]",
        Category::Logic => "bg-[#F4D63B]",
        Category::Data => "bg-[#AC01F4]",
        Category::AI => "bg-[#F46701]",
        Category::IO => "bg-[#0099D7]",
        Category::Other => "bg-[#F40A0B]",
    }
}
