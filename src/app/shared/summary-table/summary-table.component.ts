import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-summary-table',
  templateUrl: './summary-table.component.html',
  styleUrls: ['./summary-table.component.scss']
})
export class SummaryTableComponent implements OnInit {
  @Input('rows') rows;
  @Input('name') name;
  isSummary: boolean;
  isSkewness: boolean;
  objectKeys = Object.keys;
  columns: any;
  describeRows: any;
  constructor() { }

  ngOnInit() {

    this.columns = this.rows['columns'];
    this.describeRows = this.rows['rows'];
    this.isSummary = this.summary;
    this.isSkewness = this.skewness;
  }

  get summary() {
    return this.name == 'summary';
  }

  get skewness() {
    return this.name == 'skewnessAndKurtosis';
  }


}
