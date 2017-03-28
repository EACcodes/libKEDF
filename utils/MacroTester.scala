/*#!/bin/sh
exec scala "$0" "$@"
!#*/

import java.io.FileWriter
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.Files.copy
import java.nio.file.Paths.get
import sys.process._
import scala.language.postfixOps
import scala.language.implicitConversions
import scala.io.Source
import scala.math.abs

object MacroTester {

  // both of these thresholds have more to do with default printing formats than accuracy
  val NUMTHRESHE = 1e-5
  val NUMTHRESHG = 1e-5
  val DEBUG = false

  implicit def toPath (filename: String) = get(filename)

  def main(args: Array[String]): Unit = {

    val densityTrajFile = args(0)
    val potenTrajRefFile = args(1)
    val inputStub = args(2)
    val clientPath = args(3)
    
    var densities = 0
    var failuresE = 0
    var failuresEG = 0
    var failuresG = 0

    val densFileDat = Source.fromFile(densityTrajFile).getLines
    val potFileDat = Source.fromFile(potenTrajRefFile).getLines

    // copy stub file and append density location and job types
    val eInpFile = "test_e.inp"
    val gInpFile = "test_g.inp"
    copy (inputStub, eInpFile, REPLACE_EXISTING)
    copy (inputStub, gInpFile, REPLACE_EXISTING)

    val eInpF = new FileWriter(eInpFile,true)
    eInpF.write("JOB TYPE\n")
    eInpF.write("energy\n")
    eInpF.write("GRID FILE\n")
    eInpF.write("testgrid.data\n")
    eInpF.write("FILL GRID FROM FILE\n")
    eInpF.close

    val gInpF = new FileWriter(gInpFile,true)
    gInpF.write("JOB TYPE\n")
    gInpF.write("potential\n")
    gInpF.write("GRID FILE\n")
    gInpF.write("testgrid.data\n")
    gInpF.write("FILL GRID FROM FILE\n")
    gInpF.write("PRINT VERBOSE\n")
    gInpF.close

    while(densFileDat.hasNext){

      // get this grid into place, overriding what ever was there before
      val densFile = new FileWriter("testgrid.data",false)

      densities += 1

      // forward one
      densFileDat.next
      // parse grid dimensions
      val gridLine = densFileDat.next.trim.split("\\s+")
      val gridX = gridLine(0).toInt
      val gridY = gridLine(1).toInt
      val gridZ = gridLine(2).toInt

      val totGridPoints = gridX*gridY*gridZ

      var lineC = 0
      while (lineC < totGridPoints){
        val densLine = densFileDat.next.trim
        densFile.write(densLine + "\n")

        lineC += 1
      }

      densFile.flush
      densFile.close

      // execute code

      val eOutT = (clientPath + " " + eInpFile) !!

      if(DEBUG){
        println("DEBUG: energy output")
        println(eOutT)
      }
      val eOut = eOutT.split("\n")

      val gradOutT = (clientPath + " " + gInpFile) !!

      if(DEBUG){
        println("DEBUG: potential output")
        println(gradOutT)
      }
      val gradOut = gradOutT.split("\n")

      // read in reference and compare
      potFileDat.next
      val refE = potFileDat.next.trim.toDouble
      potFileDat.next

      // energy comparison
      var eE = -42.0
      for(line <- eOut){
        if(line.trim.startsWith("Energy:")){
          eE = line.split("\\s+")(1).toDouble
        }
      }

      var eG = +42.0
      for(line <- gradOut){
        if(line.trim.startsWith("Energy:")){
          eG = line.split("\\s+")(1).toDouble
        }
      }

      if(abs(eE - refE) > NUMTHRESHE){
        println("wrong energy (I): " + eE + " vs " + refE + ", diff " + abs(eE-refE))
        failuresE += 1
//        return
      } else {
        println("Energy ok from e")
      }

      if(abs(eG - refE) > NUMTHRESHE){
        println("wrong energy (II): " + eG + " vs " + refE + ", diff " + abs(eG-refE))
        failuresEG += 1
//        return
      } else {
        println("Energy ok from g")
      }

      lineC = 0
      var c = 0
      for(line <- gradOut){
        if(line.trim.startsWith("Potential:")){
          lineC += c+1
        }
        c += 1
      }

      val potent = Array.ofDim[Double](gridX,gridY,gridZ)
      for(z <- 0 until gridZ){ // slice
        lineC += 1 // the [cube ] part
        for(x <- 0 until gridX){
          val potD = gradOut(lineC).trim.split("\\s+")
          lineC += 1
          for(y <- 0 until gridY){
            // get the actual one
            val actPot = potD(y).toDouble
            potent(x)(y)(z) = actPot
          }
        }
        lineC += 1 // the empty line
      }

      for(z <- 0 until gridZ){ // slice
        for(y <- 0 until gridY){
          for(x <- 0 until gridX){
            val refPot = potFileDat.next.trim.toDouble
            val actPot = potent(x)(y)(z)
            if(abs(actPot - refPot) > NUMTHRESHG){
              println("wrong potential: " + actPot + " vs " + refPot + ", diff " + abs(actPot-refPot))
//              return
              failuresG += 1
            } 
          }
        }
      }
    }

    println("Tested " + densities + " densities.")
    println("Energy failures (e) " + failuresE)
    println("Energy failures (e+p) " + failuresEG)
    println("Potential failures " + failuresG)

  }
}
