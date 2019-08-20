import { Component, OnInit, EventEmitter, Output } from '@angular/core';
import { AuthService } from '../../services/auth.service';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router } from '@angular/router';
import { CommonService } from 'src/app/services/common.service';

@Component({
  selector: 'app-navigation',
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.scss']
})
export class NavigationComponent implements OnInit {
  navLinks: any = [];
  isLoggedIn: boolean;
  user: any;
  @Output() sideNavEvent = new EventEmitter();
  sideNavClicked = false;
  constructor(private authService: AuthService, private afauth: AngularFireAuth, private router: Router, private commonService: CommonService) { }

  ngOnInit() {
    this.navLinks = [{ path: '/home', label: 'Home' },
    { path: '/preview', label: 'Descriptive Statistics' },
    { path: '/feature-selection', label: 'Feature Engineering' },
    { path: '/result', label: 'Result' },
   // { path: '/history', label: 'History' }
    ];


    this.afauth.authState.subscribe((user) => {
      if (user && user.uid) {
        this.user = user;
        this.isLoggedIn = true;
        return;
      }
      this.isLoggedIn = false;
    });

  }

  logout() {
    this.authService.signOut().subscribe((data) => {
      if (data) {
        this.router.navigate(['/login']);
        this.isLoggedIn = false;
        this.showSideNav();
        localStorage.removeItem('targetColumn');
      }
    });

  }

  showSideNav() {
    this.sideNavClicked = !this.sideNavClicked;
    this.sideNavEvent.next(this.sideNavClicked);
  }
}
