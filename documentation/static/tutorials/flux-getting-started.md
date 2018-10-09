# Get Started with Flux
Flux is InfluxData's new data language designed for querying, analyzing, and acting on data stored in InfluxDB.
Its takes the power of InfluxQL and the TICKscript and combines them into a single, unified data scripting language.

## Setup Flux from scratch
Setting up the TICK stack with Flux is pretty simple.
If you're using the InfluxDB Sandbox, it's even simpler.
**For instructions on using Flux with the sandbox, [skip down](#using-flux-with-the-sandbox)**.

### Install Chronograf and InfluxDB Nightlies
Download the nightly builds Chronograf, and InfluxDB to get the most recent versions that include Flux and Flux-dependent
functionality.
Nightly builds for each are available on the [InfluxData Downloads](https://portal.influxdata.com/downloads) page.
Once downloaded and unpackaged, move the binaries into your system's `$PATH`.

### Configure the storage service on InfluxDB
InfluxDB stores its data within a [Time-Structured Merge Tree (TSM)](https://docs.influxdata.com/influxdb/latest/concepts/storage_engine/) format. However, there are two potential indicies
with the underlying TSM file format.  The default `index-version` is `inmem` or in-memory. Upon startup, the TSM files are
read and the in-memory index is rebuilt.   

There second option is to use the [Time Series Index (TSI)](https://docs.influxdata.com/influxdb/latest/concepts/tsi-details/) engine, which stores indexed data on disk.

If you are starting from scracth, update your the `index-version` setting under the `[data]` section in your
`influxdb.conf` to use `tsi1`:

```toml
# ...
[data]
  # ...
  index-version = "tsi1"
  # ...
```

If you already have been running InfluxDB and have data that you wish to maintain and use, you need to convert your existing
TSM-based shards to TSI-supported shards.

Use `influx_inspect buildtsi` for converting your TSM-based shards to TSI-based shards. You can read more about using [building TSI here.](https://docs.influxdata.com/influxdb/v1.6/tools/influx_inspect/#buildtsi)

### Start InfluxDB and Chronograf
InfluxDB and Chronograf are all run as separate daemonized processes and must be started separately.
Run each of the following commands in their own terminal sessions.

```bash
# Start the influxd daemon
influxd -config path/to/influxdb.conf
```

```bash
# Start Chronograf
chronograf
```

### Configure Chronograf
Open Chronograf in your browser of choice at [localhost:8888](http://localhost:8888).

Select the **wrench** icon in the left-hand navigation bar which is the **Configuration** option.
The following screen should appear:

![Configuration](../images/configure-chronograf.png)

#### Connect to InfluxDB
If not already connected to InfluxDB, you will be prompted for connection details.
Provide the necessary credentials and save.

![Connect Chronograf to InfluxDB](../images/connect-to-influxdb.png)

#### Connect to Flux
To connect Chronograf to the Flux engine within the InfluxDB OSS instance, click on the Flux Editor icon in the left navigation.

![Connect Chronograf to Flux](../images/connect-to-flux.png)

Ensure the URL of the InfluxDB OSS instance is used -- and append that with the `/v2` suffix.
For example, if you are running InfluxDB on your local machine and using `http://localhost:8086` to interact with InfluxDB,
the Flux URL should be: `http://influxdb:8086/v2`.

Once the connection is established, the **Flux Editor** is available from the **Data Explorer** and when defining cells within a Dashboard. Keep in mind that BOTH InfluxQL and Flux can be used within InfluxDB 1.7.

## Using Flux with the Sandbox
To use Flux with the [InfluxDB Sandbox](https://github.com/influxdata/sandbox),
start the sandbox with the `-nightly` flag to pull the nightly builds of Influx services.

```bash
./sandbox up -nightly
```

## Get started with the Flux Editor
The Flux Editor makes working with Flux a visual process. It consists of 3 panes:

1. **[The Script Editor](#script-editor)** Where the actual Flux code is written and displayed.
2. **[The Flux Builder](#flux-builder)** A visual representation of your Flux script used to visualize and build your script.
3. **[The Schema Explorer](#schema-explorer)** Allows you to explore the actual structure of your data as you're building Flux scripts.

![Flux Editor](../images/flux-editor.png)

Each pane can be minimized, maximized, or closed depending on how you want to work.

### Script Editor
Flux queries are written in the "Script" pane of the Flux Editor.
You can also use the [Flux Builder](#flux-builder) to visually build out queries.
As queries are updated in the Flex Builder, the are updated in the script editor.

![Script Editor](../images/flux-editor-script.png)

### Schema Explorer
The "Explore" pane of the Flux Editor allows you to visual explore the structure of your data.
This is incredibly helpful as you're building out Flux queries.

![Schema Explorer](../images/flux-editor-explore.png)

### Flux Builder
The "Build" pane is a visual representation of your Flux script that used to both visualize and build your script.
As queries are updated in the Flex Builder, the are updated in the script editor.
At any point in your Flux query, you can use the `yield()` function to visualize the current state of your query.

![Flux Builder](../images/flux-editor-build.png)

## Learn the basics of the Flux language
Flux draws inspiration from programming languages such as Lisp, Elixir, Elm,
Javascript and others, but is specifically designed for analyzing and acting on data.
For an introduction into the Flux syntax, view the
[Basic Syntax](https://github.com/influxdata/platform/blob/master/query/README.md#basic-syntax)
section in the Flux project `README`.

You can also [explore a walkthrough of the language including a handful of simple expressions and what they mean.](https://github.com/influxdata/platform/blob/nc-training/TRAIN.md#learning-flux)

## Additional Information
[Flux Introduction Slides](https://speakerdeck.com/pauldix/flux-number-fluxlang-a-new-time-series-data-scripting-language)  
[Flux Readme](https://github.com/influxdata/platform/blob/master/query/README.md)  
[Flux Specification](https://github.com/influxdata/platform/blob/master/query/docs/SPEC.md)  
