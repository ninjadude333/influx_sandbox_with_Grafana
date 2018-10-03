# influx_sandbox_with_Grafana
the official influxDB sandbox with Grafana added


# TICK Sandbox

This repo is a quick way to get the entire TICK Stack spun up and working together. It uses [Docker](https://www.docker.com/) to spin up the full TICK stack in a connected fashion. This is heavily tested on Mac and should mostly work on linux and Windows.

To get started you need a running docker installation. If you don't have one, you can download Docker for [Mac](https://www.docker.com/docker-mac) or [Windows](https://www.docker.com/docker-windows), or follow the installation instructions for Docker CE for your [Linux distribution](https://docs.docker.com/engine/installation/#server).

### Running 

To run the `sandbox`, simply use the convenient cli:

```bash
$ ./sandbox
sandbox commands:
  up           -> spin up the sandbox environment (add -nightly to grab the latest nightly builds of InfluxDB and Chronograf)
  down         -> tear down the sandbox environment
  restart      -> restart the sandbox
  influxdb     -> attach to the influx cli
  
  enter (influxdb||kapacitor||chronograf||telegraf||grafana) -> enter the specified container
  logs  (influxdb||kapacitor||chronograf||telegraf||grafana) -> stream logs for the specified container
  
  delete-data  -> delete all data created by the TICK Stack
  docker-clean -> stop and remove all running docker containers
  rebuild-docs -> rebuild the documentation container to see updates
```

To get started just run `./sandbox up`. You browser will open two tabs:

- `localhost:xxxx` - Chronograf's address. You will use this as a management UI for the full stack
- `localhost:xxxx` - Documentation server. This contains a simple markdown server for tutorials and documentation.

.env the container ports can be configured in the .env file,
also it is possible to run many instances of this on the same server as long as you run it from a different directory and make sure the ports are not overlapping.