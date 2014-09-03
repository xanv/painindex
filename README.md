## About this application
The Pain Index is a web application that builds a comprehensive index of sting pain intensity from user reports.

## Getting started
See [Getting Started](GettingStarted.md) for info on how to set everything up on your machine.
Also, be sure you have installed all the dependencies mentioned in the section below.

### Dependencies

*Sass*

This application uses Sass, which is built in [Ruby](https://www.ruby-lang.org/en/).  We also suggest you install [RVM](http://rvm.io/) and [rubygems](https://rubygems.org/) to manage your gems and enviornments.

*Graphviz*

We also use [Graphviz](http://www.graphviz.org/), which automatically generates a schema diagram from the models.  To set up Graphviz, [install](http://www.graphviz.org/Download..php) it on your computer.

*Other Dependencies*

The rest of the dependencies are fairly standard for a Django application. You can review them in the [requirements.txt](requirements.txt) file.  To install all the requirements, run ``pip install -r requirements.txt``.  If it fails, you might need to install Graphviz first (see above).

### Setting up the database
The database is currently a standard PostgreSQL database that requires no special setup.  However, in the future this section will contain notes on how to update statistics from pain reports and any other non-standard procedures regarding the database.

## Running the test suite
Again, the test suite is currently standard and requires no special setup or comment.


## Git Workflow

### To start working on the app:
1. ``git clone https://github.com/xanv/painindex.git``
2. ``git checkout -b "your_branch_name"``

### Naming branches
* The production branch is ``master``.
* The development branch is ``develop``. 
* Feature branches branch off from ``develop`` and are named ``feature-<description>``.  
* Hotfix branches are named ``hotfix-<description>``;  they may branch from develop or master and should be merged back into both.

### To submit a pull request:
1. ``git checkout {your_branch_name}``
2. Add and commit your changes
3. ``git pull origin master`` or ``git pull origin develop`` (will try to merge; fix conflicts if necessary)
4. Add and commit any changes
5. ``git push origin {your_branch_name}``
6. On GitHub, submit a pull request

## Sass

This application uses Sass.  All Sass files should be kept in ``painindex/static/painindex/sass``.  They can be compiled to ``painindex/static/painindex/css`` via the following command from the root directory:
```
sass --update painindex/static/painindex/sass:painindex/static/painindex/css
```

To have the css files update automatically while running a development server, run the following command from the root directory in a separate shell:
```
sass --watch painindex/static/painindex/sass:painindex/static/painindex/css
```

## License information

This application is free to use under the terms of the [MIT License](LICENSE).