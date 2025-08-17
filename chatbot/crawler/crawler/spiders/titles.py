import scrapy
from urllib.parse import urljoin

class TitlesSpider(scrapy.Spider):
    name = "titles"
    allowed_domains = ["www.gov.br", "gov.br"]
    start_urls = ["https://www.gov.br/pt-br"]

    def parse(self, response):
        # títulos da página atual
        for t in response.css("a::text, h1::text, h2::text").getall():
            t = t.strip()
            if t:
                yield {"url": response.url, "title": t}

        # segue até 20 links internos (para não exagerar no bloco 1)
        count = 0
        for href in response.css("a::attr(href)").getall():
            if count >= 20:
                break
            href = href.strip()
            if href and href.startswith("/"):
                count += 1
                yield response.follow(href, callback=self.parse)
