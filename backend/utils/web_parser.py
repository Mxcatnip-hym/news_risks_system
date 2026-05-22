from newspaper import Article


def get_news_text(url: str) -> str:
    """
    从 URL 抓取新闻正文
    依赖：newspaper3k
    """
    article = Article(url, language="zh")
    article.download()
    article.parse()
    title = article.title or ""
    body = article.text or ""
    return (title + " " + body).strip()
