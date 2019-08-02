import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, FormControl, Validators } from '@angular/forms';
import { Subscription } from 'rxjs';
import { AuthService } from '../../services/auth.service';
import { MatSnackBar } from '@angular/material';
import { Router } from '@angular/router';

@Component({
  selector: 'app-reset-password',
  templateUrl: './reset-password.component.html',
  styleUrls: ['./reset-password.component.scss']
})
export class ResetPasswordComponent implements OnInit {

  resetForm: FormGroup;
  subscription: Subscription[] = [];
  resetEmail: string;
  constructor(private fb: FormBuilder, private authService: AuthService, private snackBar: MatSnackBar, private router: Router) { }

  ngOnInit() {
    this.createForm();
  }

  createForm() {
    this.resetForm = this.fb.group({
      email: new FormControl('', {
        validators: Validators.compose([
          Validators.required,
          Validators.pattern('^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
        ]), updateOn: 'change'
      })
    });
  }

  submit() {
    this.authService.passwordReset(this.resetForm.value.email)
      .subscribe((data) => {
        if (data === 'success') {
          this.snackBar.open('Reset link sent successfully to your email. ', 'Close', { duration: 3000 });
          this.router.navigate(['login']);
        }
        if (data) {
          switch (data.code) {
            case 'auth/invalid-email': {
              this.snackBar.open(data.message, 'Close', { duration: 3000 });
              break;
            }
            case 'auth/user-not-found': {
              this.snackBar.open('No User Found', 'Close', { duration: 3000 });
              break;
            }
          }
        }
      });
  }

}
