import { Component, OnInit, OnDestroy } from '@angular/core';
import { FormGroup, FormBuilder, FormControl, Validators } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { MatSnackBar } from '@angular/material';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { AlertsService } from '../../services/alert.service';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.scss']
})
export class SignupComponent implements OnInit, OnDestroy {
  signupForm: FormGroup;
  subscription: Subscription[] = [];
  nickname: string;

  constructor(private fb: FormBuilder, private authService: AuthService,
    private snackBar: MatSnackBar, private router: Router, private alertService: AlertsService) {
  }

  ngOnInit() {
    this.createForm();
  }


  createForm() {
    this.signupForm = this.fb.group({
      nickname: new FormControl('', { validators: [Validators.required, Validators.minLength(4)], updateOn: 'change' }),
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
      this.authService.signup(this.signupForm.value.nickname, this.signupForm.value.email, this.signupForm.value.password)
        .subscribe(
          (data) => {
            console.log(data);
            if (data) {
              if (data.code) {
                this.snackBar.open(data.message, 'Close', { duration: 3000 });
              }
              if (data.user) {
                this.router.navigate(['home']);
                this.snackBar.open('Congrats! Account Created', 'Close', { duration: 3000 });
              }
            }
          })
    );
  }

  signupWithGmail() {
    this.authService.signupWithGmail().subscribe((data) => {
      if (data && data.user) {
        this.router.navigate(['home']);
        this.snackBar.open('Congrats! Account Created', 'Close', { duration: 3000 });
      }
      if (data && data.code) {
        this.alertService.setAlertsForAuthProviderLogins(data);
      }

    });
  }

  signupWithGithub() {
    this.authService.signupWithGithub().subscribe((data) => {
      if (data && data.user) {
        this.router.navigate(['home']);
        this.snackBar.open('Congrats! Account Created', 'Close', { duration: 3000 });
      }
      if (data && data.code) {
        this.alertService.setAlertsForAuthProviderLogins(data);
      }
    });
  }



  ngOnDestroy() {
    this.subscription.forEach(sub => sub.unsubscribe());
  }


}
