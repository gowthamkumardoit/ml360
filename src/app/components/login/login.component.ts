import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators, FormControl } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { MatSnackBar, MatDialog } from '@angular/material';
import { Subscription, Observable } from 'rxjs';
import { Router } from '@angular/router';
import { AlertsService } from '../../services/alert.service';
import { AngularFirestore } from '@angular/fire/firestore';
import { map } from 'rxjs/operators';
import { User } from '../../interfaces/user';
import { AngularFireAuth } from '@angular/fire/auth';
@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {

  loginForm: FormGroup;
  subscription: Subscription[] = [];
  resetEmail: string;

  constructor(private fb: FormBuilder, private authService: AuthService,
    private snackBar: MatSnackBar, private router: Router,
    private alertsService: AlertsService,
    private db: AngularFirestore, private afauth: AngularFireAuth) { }

  ngOnInit() {
    this.createForm();
  }

  createForm() {
    this.loginForm = this.fb.group({
      password: new FormControl('', { validators: [Validators.required, Validators.minLength(6)], updateOn: 'change' }),
      email: new FormControl('', {
        validators: Validators.compose([
          Validators.required,
          Validators.pattern('^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
        ]), updateOn: 'change'
      })
    });
  }

  submit() {
    this.subscription.push(
      this.authService.login(this.loginForm.value.email, this.loginForm.value.password).subscribe((data) => {
        console.log(data);
        if (data && data.user) {
          this.router.navigate(['/home']);
        }
        if (data && data.code) {
          this.alertsService.setAlertForLogin(data);
        }
      }
      )
    );
  }


  loginWithGmail() {
    this.subscription.push(
      this.authService.loginWithGmail().subscribe((data) => {
        if (data && data.user) {
          this.router.navigate(['/home']);
        }
        if (data && data.code) {
          this.alertsService.setAlertsForAuthProviderLogins(data);
        }
      })
    );
  }

  loginWithGithub() {
    this.subscription.push(
      this.authService.loginWithGithub().subscribe((data) => {
        if (data && data.user) {
          this.router.navigate(['/home']);
        }
        if (data && data.code) {
          this.alertsService.setAlertsForAuthProviderLogins(data);
        }
      })
    );
  }

}
