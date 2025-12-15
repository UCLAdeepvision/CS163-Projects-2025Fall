# Use official Ruby image with version 3.1.4
FROM ruby:3.1.4-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Gemfile and Gemfile.lock (if exists)
COPY Gemfile* ./

# Install bundler and gems
RUN gem install bundler:2.5.23 && \
    bundle install

# Copy the rest of the application
COPY . .

# Expose port 4000 for Jekyll server
EXPOSE 4000

# Default command to run Jekyll server
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload"]



