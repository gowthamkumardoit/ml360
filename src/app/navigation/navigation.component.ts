import { Component, OnInit } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { AngularFireAuth } from '@angular/fire/auth';

@Component({
  selector: 'app-navigation',
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.scss']
})
export class NavigationComponent implements OnInit {
  navLinks: any = [];
  isLoggedIn: boolean = false;
  constructor(private authService: AuthService, private afauth: AngularFireAuth) { }

  ngOnInit() {
    this.navLinks = [{ path: '/home', label: 'Home' },
    { path: '/preview', label: 'Preview/Summary' },
    { path: '/feature-selection', label: 'Plots/Feature Selection' },
    { path: '/result', label: 'Result' }];

    
    this.afauth.authState.subscribe((state) => {
      this.isLoggedIn = this.authService.authenticated;
    })
  }

  logout() {
    this.authService.logout();
    this.isLoggedIn = this.authService.authenticated;
  }
}
