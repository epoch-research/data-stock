from utils import *

def indexed_web_2024():
    # Based on the distribution of reported Google Search API results for a number
    # of words with known frequency on Internet text (https://www.worldwidewebsize.com/)
    # the size of Google's index is 111B web pages [60B, 543B]. Google has a good chance of
    # including all the relevant information in the clearweb
    # Internet Archive has 866B web pages, but many of them are multiple captures over time of
    # the same page
    # CommonCrawl contains about 250B web pages, 75B of them have unique URLs
    number_of_unique_indexed_websites = sq.to(75e9, 1e12, credibility=95)

    # Estimated for CommonCrawl
    average_bytes_per_website = sq.to(6100, 8200, credibility=95)

    # A distribution with median around 4 and 95% CI of [2,5]
    # This range was observed for multiple tokenizers and text distributions
    average_bytes_per_token = 6-sq.to(1,4,credibility=95)

    total_stock = number_of_unique_indexed_websites * average_bytes_per_website / average_bytes_per_token

    return total_stock


def stock_indexed_web(t):
    # Calculated from CommonCrawl between 2013 and 2021
    average_webpage_growth = sq.to(1.02,1.04,credibility=95)

    # Unclear, but for Google's index size to remain constant the web has to at least produce enough
    # new pages to offset link rot. Since estimates of link rot are often a single-digit percentage
    # per year, I use 0%-10%
    growth_in_number_of_webpages = sq.to(1,1.1,credibility=95)

    return indexed_web_2024() * (average_webpage_growth*growth_in_number_of_webpages)**(t-2024)


if __name__ == "__main__":
    print(np.quantile(indexed_web_2024() @ 100_000, 0.025) / 1e12)
    print(np.quantile(indexed_web_2024() @ 100_000, 0.5) / 1e12)
    print(np.quantile(indexed_web_2024() @ 100_000, 0.975) / 1e12)
