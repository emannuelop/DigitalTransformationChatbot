import scrapy

class TitlesSpider(scrapy.Spider):
    name = "titles"
    allowed_domains = ["www.gov.br", "gov.br"]
    start_urls = ["https://www.gov.br/pt-br"]

    def parse(self, response):
        # pega textos visíveis de links e cabeçalhos, limpa e emite
        texts = response.css("a::text, h1::text, h2::text").getall()
        for t in texts:
            t = (t or "").strip()
            if t:
                yield {"title": t}
