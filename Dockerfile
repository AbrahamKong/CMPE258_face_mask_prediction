#Base Image to use
FROM python:3.7

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PYTHONUNBUFFERED True

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

# #Copy all files in current directory into app directory
# COPY . /app

# #Change Working Directory to app directory
# WORKDIR /app

# #Run the application on port 8080
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD streamlit run --server.port 8080 app.py
