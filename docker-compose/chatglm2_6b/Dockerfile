# 使用基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制当前目录下的环境文件到容器中
COPY . .

# 创建conda环境并安装依赖
RUN pip install -r requirements.txt

CMD [ "python", "scrapy.py" ]