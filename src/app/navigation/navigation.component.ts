import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-navigation',
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.scss']
})
export class NavigationComponent implements OnInit {
  navLinks: any = [];
  constructor() { }

  ngOnInit() {
    this.navLinks = [{ path: '/home', label: 'Home' },
    { path: '/preview', label: 'Preview/Summary' },
    { path: '/feature-selection', label: 'Plots/Feature Selection' },
    { path: '/result', label: 'Result' }];
  }

}
