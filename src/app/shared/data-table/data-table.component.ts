import { Component, OnInit, ViewChild } from '@angular/core';
import { MatPaginator } from '@angular/material/paginator';
import { MatSort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';
import { PreviewService } from 'src/app/services/preview.service';


@Component({
  selector: 'app-data-table',
  templateUrl: './data-table.component.html',
  styleUrls: ['./data-table.component.scss']
})
export class DataTableComponent implements OnInit {

  constructor(private previewService: PreviewService) { }
  displayedColumns: string[] = ['name', 'delimiter', 'date'];
  data;
  dataSource;

  @ViewChild(MatPaginator, { static: true }) paginator: MatPaginator;
  @ViewChild(MatSort, { static: true }) sort: MatSort;

  ngOnInit() {
    this.getFilesList().then((res) => {
      if (res) {
        this.dataSource = new MatTableDataSource(this.data);
        this.dataSource.sort = this.sort;
        this.dataSource.paginator = this.paginator;
      }
    });

  }

  getFilesList() {
    return new Promise((resolve) => {
      this.previewService.getFilesForUsers().subscribe((data) => {
        if (data.length > 0) {
          this.data = [];
          data.forEach((ele: any, i: number) => {
            this.data.push({ ...ele, no: i + 1 });
          });
          resolve(true);
        } else {
          resolve(false);
        }
      });
    });
  }
}

export interface PeriodicElement {
  name: string;
  position: number;
  weight: number;
  symbol: string;
}





