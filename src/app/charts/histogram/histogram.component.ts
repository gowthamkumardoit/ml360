import { Component, OnInit, Input, AfterViewInit } from '@angular/core';
import * as d3 from 'd3';
@Component({
  selector: 'app-histogram',
  templateUrl: './histogram.component.html',
  styleUrls: ['./histogram.component.scss']
})
export class HistogramComponent implements OnInit, AfterViewInit {
  @Input('dataSource') dataSource;

  constructor() { }

  ngOnInit() {

  }

  ngAfterViewInit(): void {
    //Called after ngAfterContentInit when the component's view has been initialized. Applies to components only.
    //Add 'implements AfterViewInit' to the class.
    // var chartDiv = document.getElementById("chart1");
    // let data = d3.range(1000).map(d3.randomNormal(20, 5));
    // data = data.sort();
    // console.log(data);
    // let formatCount = d3.format(",.0f");

    // let svg = d3.select(chartDiv).append("svg"),
    //   margin = { top: 10, right: 30, bottom: 30, left: 30 },
    //   width = 960 - margin.left - margin.right,
    //   height = 500 - margin.top - margin.bottom,
    //   g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // let x = d3.scaleLinear()
    //   .rangeRound([0, width]);

    // let bins = d3.histogram()
    //   .thresholds(x.domain())
    //   (data.sort());

    // let y = d3.scaleLinear()
    //   .domain([0, d3.max(bins, (d: any) => { return parseInt(d.length); })])
    //   .range([height, 0]);

    // let bar = g.selectAll(".bar")
    //   .data(bins)
    //   .enter().append("g")
    //   .attr("class", "bar")
    //   .attr("transform", (d: any) => { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

    // bar.append("rect")
    //   .attr("x", 1)
    //   .attr("width", x(bins[0].x1) - x(bins[0].x0) - 1)
    //   .attr("height", (d: any) => { return height - y(d.length); });

    // bar.append("text")
    //   .attr("dy", ".75em")
    //   .attr("y", 6)
    //   .attr("x", (x(bins[0].x1) - x(bins[0].x0)) / 2)
    //   .attr("text-anchor", "middle")
    //   .text((d: any) => { return formatCount(d.length); });

    // g.append("g")
    //   .attr("class", "axis axis--x")
    //   .attr("transform", "translate(0," + height + ")")
    //   .call(d3.axisBottom(x));

  }

}
