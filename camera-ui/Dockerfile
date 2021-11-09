FROM node:latest as build

WORKDIR /app
COPY package*.json /app/
RUN npm install
COPY ./ /app/
RUN npm run build

FROM nginx:latest as serve

COPY --from=build /app/build/ /var/www
COPY ./nginx.conf /etc/nginx/nginx.conf
EXPOSE 80