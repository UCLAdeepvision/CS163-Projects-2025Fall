# UCLA DLCV Course Project

Project page: https://ucladeepvision.github.io/CS163-Projects-2025Fall/


## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version 3.1.4 

For MacOS:
```
brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
rbenv install 3.1.4 && rbenv global 3.1.4
```
For Ubuntu: 
```
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
rbenv install 3.1.4 && rbenv global 3.1.4
```

Check your Ruby version
```
ruby -v # should be 3.1.4
```

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
bundler install
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS163-Projects-2025Fall/ by default), go to this address with your browser.

## Working on the project

1. Create a folder with your team id under ```./assets/images/your-teamid```, you will use this folder to store all the images in your project.

2. Copy the template at ```./_posts/2025-01-01-team00-instruction-to-post.md``` and rename it with format "yyyy-mm-dd-teamXX-projectshortname.md" under ```./_posts/```, for example, **2025-01-01-team01-object-detection.md**

3. Check out the sample post we provide at https://ucladeepvision.github.io/CS163-Projects-2025Fall/ and the source code at https://raw.githubusercontent.com/UCLAdeepvision/CS163-Projects-2025Fall/main/_posts/2025-01-01-team00-instruction-to-post.md as well as basic Markdown syntax at https://www.markdownguide.org/basic-syntax/

4. Start your work in your .md file. You may **only** edit the .md file you just copied and renamed, and add images to ```./assets/images/your-teamid```. *Please do NOT change any other files in this repo.*

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.

## Deadlines  
You should update your final blog post by making a pull request on TODO. 

-----

Kudos to [Tianpei](https://gutianpei.github.io/), who originally developed this site for CS 188 in Winter 2022.