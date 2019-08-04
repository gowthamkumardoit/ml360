import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-side-nav',
  templateUrl: './side-nav.component.html',
  styleUrls: ['./side-nav.component.scss']
})
export class SideNavComponent implements OnInit {
  items: any[] = [];
  constructor() { }

  ngOnInit() {
    this.items = [
      { id: 1, name: 'home', url: '/home' },
      { id: 2, name: 'assignment', url: '/preview' },
      { id: 3, name: 'assessment', url: '/feature-selection' },
      { id: 4, name: 'dvr', url: '/result' },
      { id: 5, name: 'settings', url: '/settings' },
      { id: 6, name: 'history', url: '/history' },
    ];
  }

  getSideNavClickEvent(event) {
    if (event) {
      this.openNav();
      return;
    }
    this.closeNav();
  }
  /* Set the width of the side navigation to 100px */
  openNav() {
    document.getElementById('mySidenav').style.width = '100px';
    document.getElementById('main').style.marginLeft = '100px';
  }

  /* Set the width of the side navigation to 0 */
  closeNav() {
    document.getElementById('mySidenav').style.width = '0';
    document.getElementById('main').style.marginLeft = '0';
  }
}
