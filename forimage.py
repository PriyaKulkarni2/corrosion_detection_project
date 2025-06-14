from icrawler.builtin import BingImageCrawler

def download_images(keyword, num_images, directory):
    crawler = BingImageCrawler(storage={"root_dir": directory})
    crawler.crawl(keyword=keyword, max_num=num_images)

download_images("corroded metal surface", 100, "./data/corrosion")
download_images("clean metal surface", 100, "./data/no_corrosion")
