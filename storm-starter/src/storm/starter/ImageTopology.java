package storm.starter;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import backtype.storm.utils.Utils;


import backtype.storm.spout.SpoutOutputCollector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.*;
import java.util.List;

/**
 * This is a basic example of a Storm topology.
 */
public class ImageTopology {

  public static class TestImageSpout extends BaseRichSpout {
     // public static Logger LOG = LoggerFactory.getLogger(TestImageSpout.class);
      boolean _isDistributed;
      SpoutOutputCollector _collector;

      public TestImageSpout() {
          this(true);
      }

      public TestImageSpout(boolean isDistributed) {
          _isDistributed = isDistributed;
      }

      public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
          _collector = collector;
      }

      public void close() {}

      //////mover

      static final File dir = new File("/vagrant/storm-starter/src/storm/starter/images");

      static final String[] EXTENSIONS = new String[]{
              "gif", "png", "jpg"
      };

      static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

          @Override
          public boolean accept(final File dir, final String name) {
              for (final String ext : EXTENSIONS) {
                  if (name.endsWith("." + ext)) {
                      return (true);
                  }
              }
              return (false);
          }
      };
      ////////

      public void nextTuple() {
          Utils.sleep(100);
//          final String[] words = new String[] {"nathan", "mike", "jackson", "golda", "bertels"};
//          final Random rand = new Random();
//          final String word = words[rand.nextInt(words.length)];
          ///////
          if (dir.isDirectory()) {
              for (final File f : dir.listFiles(IMAGE_FILTER)) {
                  BufferedImage img = null;

                  try {
                      img = ImageIO.read(f);
                      _collector.emit(new Values(img));
                  } catch (final IOException e) {

                  }
              }
          }
          ////////

      }

      public void ack(Object msgId) {}
      public void fail(Object msgId) {}

      public void declareOutputFields(OutputFieldsDeclarer declarer) {
          declarer.declare(new Fields("img"));
      }

      @Override
      public Map<String, Object> getComponentConfiguration() {
          if(!_isDistributed) {
              Map<String, Object> ret = new HashMap<String, Object>();
              ret.put(Config.TOPOLOGY_MAX_TASK_PARALLELISM, 1);
              return ret;
          } else {
              return null;
          }
      }

  }

  public static class ImageBolt extends BaseRichBolt {
    OutputCollector _collector;

    @Override
    public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
      _collector = collector;
    }

    @Override
    public void execute(Tuple tuple) {
//      _collector.emit(tuple, new Values(tuple.getString(0) + "!!!"));

       // java.awt.image.BufferedImage

        BufferedImage img = (BufferedImage) tuple.getValueByField("img");


       //_collector.emit(tuple, tuple.getValues());
        System.out.println(" width : " + img.getWidth());
        System.out.println(" height: "+img.getHeight());
      _collector.ack(tuple);
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
      declarer.declare(new Fields("img"));
    }


  }

  public static void main(String[] args) throws Exception {
    TopologyBuilder builder = new TopologyBuilder();

    builder.setSpout("image", new TestImageSpout(), 10);
    builder.setBolt("exclaim1", new ImageBolt(), 3).shuffleGrouping("image");
    builder.setBolt("exclaim2", new ImageBolt(), 2).shuffleGrouping("exclaim1");

    Config conf = new Config();
    conf.setDebug(true);

    if (args != null && args.length > 0) {
      conf.setNumWorkers(3);

      StormSubmitter.submitTopology(args[0], conf, builder.createTopology());
    }
    else {

      LocalCluster cluster = new LocalCluster();
      cluster.submitTopology("test", conf, builder.createTopology());
      Utils.sleep(10000);
      cluster.killTopology("test");
      cluster.shutdown();
    }
  }
}
