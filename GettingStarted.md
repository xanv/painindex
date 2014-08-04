Instructions for getting started:
(Untested, preliminary/incomplete guide)


0. Ensure you have python 2.7, git and postgresql installed on your system.

1. Clone painindex to your machine.

2. Install pip if you don't have it already.

3. Install and configure virtualenvwrapper.  This will enable us to create a virtual environment
   for development that's consistent across our machines. First, install:
    ``pip install virtualenvwrapper``

   a) Make a folder for your virtual environments to live in, e.g.:
    mkdir ~/.virtualenvs
    
   b) Add these lines to end of .bashrc (or your OS equivalent):
    export WORKON_HOME=~/.virtualenvs           # or the folder you just created
    export PROJECT_HOME=~/Code/Projects/dev     # or wherever you cloned the project to
    source /usr/local/bin/virtualenvwrapper.sh  # or your path to virtualenvwrapper.sh
    
   c) Reload .bashrc, e.g. with
    source .bashrc      # (or close and reopen the terminal)

4. Now create a virtualenv, which I assume is called painindex:
    mkvirtualenv -p python2.7 painindex

   This environment can be entered/exited with
       workon painindex
       deactivate

5. Install dependencies into the virtualenv.
   For what follows, make sure you are inside the virtualenv, which already contains
   python2.7 and pip.
   Navigate to the root of the painindex project. The needed packages are found in requirements.txt.
   Hopefully, this will install them in one fell swoop:
    pip install -r requirements.txt

  If you run into hangups, install the offending packages another way.
  Make sure to get the specified version.

6. Next, create your postgres database for development. Mine is named painindex, with host postgres.

7. Create a file in painindex/settings called env.py.  To it add the credentials for the db:
   ```
   DATABASES = {
      'default': {
         'ENGINE': 'django.db.backends.postgresql_psycopg2',
         'NAME': 'painindex',
         'USER': 'postgres',
         'PASSWORD': '<password for your db>',
         'HOST': '',
         'PORT': '5432',
     }
   }
   ```


   This is imported in settings_dev.py and gives Django access to your db.
8. You may need to make a slight change to your postgres configuration file, pg_hba.conf:
   Near the bottom, you will see the lines:
   ```
    # Database administrative login by Unix domain socket 
    local   all             postgres                                peer
   ```

  Change "peer" to "trust", which should be fine if your postgresql is just being used 
  locally for development.



At this point, we are done with the one-time setup.

Now and in the future, when you update your local repo, you should run
any migrations with 
  python manage.py migrate

That's it.  You should be able to launch the server by navigating
to the project directory (the dir with manage.py inside) and executing:
  python manage.py runserver
If you visit localhost:8000, you should see the site!
