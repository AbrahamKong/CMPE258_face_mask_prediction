# Heroku App Deployment Notes

## Overall Steps
1. Create Heroku account
1. Create Github account
1. Clone or Fork [this](https://github.com/coryroyce/emnist_letter_exploration_and_prediction.git) repo
1. Link Gihub account to Heroku
1. Deploy Heroku App from Github subfolder

## Clone or Fork Repo
1. Clone or Fork the emnist_letter_exploration_and_prediction to your github
2. Clone can be done through this [link](https://github.com/coryroyce/emnist_letter_exploration_and_prediction.git)

## Link Github Account to Heroku
1. Create a new app in Heroku and link your github profile. See [this YouTube video](https://www.youtube.com/watch?v=3tK9qIdoJ6I&ab_channel=GAURAVSUNILAGARWAL14BCE0127)
## Deploy Heroku App From Github Subfolder
Since this application is in a sub directory in Github, it requires the use of the a Buildpacks when deploying. 
- To install the Buildpacks for Heroku see the first half of this [YouTube](https://www.youtube.com/watch?v=gCjrEGAZCkw&t=376s&ab_channel=MikeDerycke) video.
- [Github link](https://github.com/timanovsky/subdir-heroku-buildpack) for the build package or just use this url in Heroku's build-pack https://github.com/timanovsky/subdir-heroku-buildpack.git
- Also in the Heroku settings, set the config variable  key to: PROJECT_PATH and the value to: heroku_app








