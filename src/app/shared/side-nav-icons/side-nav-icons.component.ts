import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-side-nav-icons',
  templateUrl: './side-nav-icons.component.html',
  styleUrls: ['./side-nav-icons.component.scss']
})
export class SideNavIconsComponent implements OnInit {
  @Input() iconList;
  constructor() { }

  ngOnInit() {
  }

}
