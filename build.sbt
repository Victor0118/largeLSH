import Dependencies._

val sparkVersion = "2.2.0"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "ca.uwaterloo",
      scalaVersion := "2.11.12",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "LargeLSH",
    resolvers ++= Seq(
      "apache-snapshots" at "http://repository.apache.org/snapshots/",
      "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven",
      "Repo at github.com/ankurdave/maven-repo" at "https://github.com/ankurdave/maven-repo/raw/master"
    ),
    libraryDependencies ++= Seq(
      scalaTest % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
      "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
      "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
      "org.rogach" %% "scallop" % "3.1.1",
      "org.scalanlp" %% "breeze" % "0.13.2",
      "org.scalanlp" %% "breeze-natives" % "0.13.2",
      "amplab" % "spark-indexedrdd" % "0.4.0"
    )
  )
