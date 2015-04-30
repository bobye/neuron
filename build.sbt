name := "neuron"

scalaVersion := "2.11.4"

libraryDependencies  ++= Seq(
            // other dependencies here
            "org.scalanlp" % "breeze_2.11" % "0.11.2",
            // native libraries are not included by default. add this if you want them
            // native libraries greatly improve performance, but increase jar sizes.
            "org.scalanlp" % "breeze-natives_2.11" % "0.11.2",
	    // Logging
            "org.slf4j" % "slf4j-simple" % "1.7.6",
            // STM
            "org.scala-stm" %% "scala-stm" % "0.7",
            // Nak machine learning
            // "org.scalanlp" % "nak" % "1.2.1"
	    "org.scalacheck" %% "scalacheck" % "1.11.3" % "test",
	    "org.scalatest" %% "scalatest" % "2.1.3" % "test"
)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds, use this.
            // "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
