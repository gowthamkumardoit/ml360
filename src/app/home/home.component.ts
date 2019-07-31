import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  fileName: string = 'Select the file!'
  isFileupload: boolean = false;
  isNextEnabled: boolean = false;

  selected: any;
  constructor() { }

  ngOnInit() {
  }

  changeAttr(event) {
    console.log(event);
    this.fileName = event.target.files[0].name || 'Select the file!';
    this.isFileupload = true;
    const files: FileList = event.target.files[0];

    // setTimeout(() => {
    //   this.isFileupload = false;
    //   this.isNextEnabled = true;
    // }, 10500);
    this.uploadFile(files);
  }

  uploadFile(files) {
  }

}
