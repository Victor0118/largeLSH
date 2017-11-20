import Dependencies._

val sparkVersion = "2.2.0"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "ca.uwaterloo",
      scalaVersion := "2.11.12",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "Hello",
    resolvers ++= Seq(
      "apache-snapshots" at "http://repository.apache.org/snapshots/"
    ),
    libraryDependencies ++= Seq(
      scalaTest % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    )
  )
