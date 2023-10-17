# Install dependencies only when needed
FROM node:current-alpine 
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat && npm install -g pnpm
WORKDIR /fastgpt

COPY . .

RUN ls -a

RUN cd ./projects/app && pnpm install

EXPOSE 3000

ENTRYPOINT ["sh","-c","cd ./projects/app && pnpm run dev"]