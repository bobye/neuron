name := "neuron"

scalaVersion := "2.10.3"

libraryDependencies  ++= Seq(
            // other dependencies here
            "org.scalanlp" % "breeze_2.10" % "0.7-SNAPSHOT",
            // native libraries are not included by default. add this if you want them (as of 0.7-SNAPSHOT)
            // native libraries greatly improve performance, but increase jar sizes.
            "org.scalanlp" % "breeze-natives_2.10" % "0.7-SNAPSHOT",
            "org.slf4j" % "slf4j-simple" % "1.7.6",
            "org.scala-stm" %% "scala-stm" % "0.7"
)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.7-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/releases/"
)


//snapshots:
libraryDependencies  ++= Seq(
            "org.scalanlp" %% "breeze" % "0.7-SNAPSHOT"
)
