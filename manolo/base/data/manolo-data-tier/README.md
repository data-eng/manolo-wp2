How to Use manolo_client docker image

DOCKER:

Create docker image:

Go to the project root folder (where the Dockerfile is) and run this :

Using cache (BuildKit available):

    docker build --build-arg USE_CACHE=true -t manolo_api:latest .

Without cache (No BuildKit available):

    docker build --build-arg USE_CACHE=false -t manolo_api:latest .

This makes a docker image with the name manolo_api

Export docker image:

After making the image you can export a .tar with this:

        docker save -o manolo_api.tar manolo_api:latest

Import docker image:

First download the docker image, the image's name is:

    manolo_api_latest.tar

You can then load it from another machine by doing:

        docker load -i manolo_api.tar

Next use:

    docker load -i manolo_api_latest.tar

This loads the image into your docker environment like any other image

Next you can use the image by using the docker-compose file by doing:

The command below works if you are in the same folder as the docker-compose file

    docker-compose --profile dev up

The starts the db and the api

The db is on

    port 8432 

(can be changed in the docker-compose file)

It uses postgres and has a

    name: manolo
    password: manolo

(can be changed in the docker-compose file)

The api is on

    port: 5000 

(can be changed in the docker-compose file)

In the environment you must add the DB variables that refer to the db from
above, you can also

Add the MLFLOW_URL which is the url of the mlflow server

After running the docker-compose file, you can use the api by going to

http://localhost:5000

Documentation, it is on the swagger page at

http://localhost:5000/swagger

NOTES:
By default the api creates a user with the username of manolo and the password of manolo

the api also creates:

    -predicates: "|_", "--", "->", "runs_on"
    -datastructures: "mlflow":1, "topology":2

PYTHON WRAPPER:

Be sure to install the requirements.txt file in the python_wrapper folder

The demo folder is in :

    root/python_wrapper/manolo_client/demo

There are several demos in the demo folder
each demo is a different way to use the api

Be sure to read the comments and to change the variables to your own values (ports, paths, etc.)

NOTE:
    
Due to SSL and Certificate issues, the api is not using https but http to use https 

you must use self signed certificates and configure your reverse proxy to use them