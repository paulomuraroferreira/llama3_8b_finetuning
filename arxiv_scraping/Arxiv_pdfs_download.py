import os
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

class ArxivPDFDownloader:
    def __init__(self, user_agent, download_dir):
        self.user_agent = user_agent
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.driver = self.initialize_webdriver()

    def initialize_webdriver(self):
        options = Options()
        options.set_preference("general.useragent.override", self.user_agent)
        driver = webdriver.Firefox(options=options)
        return driver

    def get_pdf_links(self, url):
        self.driver.get(url)
        links = self.driver.find_elements(By.TAG_NAME, 'a')
        link_data = [(link.get_attribute('href'), link.text) for link in links if link.get_attribute('href') and "arxiv.org/pdf" in link.get_attribute('href') and link.text == 'pdf']
        return link_data

    def download_pdfs(self, link_data):
        for url, text in link_data:
            pdf_response = requests.get(url)
            if pdf_response.status_code == 200:
                pdf_filename = os.path.join(self.download_dir, url.split('/')[-1] + '.pdf')
                with open(pdf_filename, 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"Downloaded: {pdf_filename}")
            else:
                print(f"Failed to download: {url}")

    def close(self):
        self.driver.quit()


if __name__ == "__main__":

    # Usage
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    download_dir = './pdfs'
    url = 'https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=llm&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2024-04-19&date-to_date=2024-07-04&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first'

    downloader = ArxivPDFDownloader(user_agent, download_dir)
    link_data = downloader.get_pdf_links(url)
    downloader.download_pdfs(link_data)
    downloader.close()
