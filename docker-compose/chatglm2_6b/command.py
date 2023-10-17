import aiohttp
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fake_useragent import UserAgent
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化User-Agent生成器
user_agent = UserAgent()

class SearchRequest(BaseModel):
    searchKey: str

class SearchResult(BaseModel):
    title: str
    description: str
    def __str__(self):
        return f"标题: {self.title}; 描述: {self.description}"

@app.get("/search")
async def search_bing(searchKey: str):
    # 随机生成User-Agent
    headers = {
        "User-Agent": user_agent.random
    }

    print(user_agent.random)

    # 构建Bing搜索的URL
    search_url = f"https://cn.bing.com/search?q={searchKey}"

    try:
        async with aiohttp.ClientSession() as session:
            # 发送HTTP请求获取搜索结果页面
            async with session.get(search_url, headers=headers) as response:
                # 使用BeautifulSoup解析HTML页面
                responseText = await response.text()
                soup = BeautifulSoup(responseText, 'html.parser')

                # 提取搜索结果的标题和描述
                results = soup.find_all('li', class_='b_algo')

                search_results = []
                for result in results:
                    title = result.find('h2').text
                    description = result.find('p').text
                    search_results.append(SearchResult(title=title, description=description).__str__())
                if not search_results:
                    outcome = '\n'.join(search_results)
                    return {"prompt": f'下文是必应搜索的结果，你可以提供实时信息，根据搜索结果回答问题。搜索词: {searchKey}; 必应搜索结果:\n {outcome}'}
                else:
                    return {"prompt": ''}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8007, workers=1)
