import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HomeComponent } from './components/home/home.component';
import { PreviewComponent } from './components/preview/preview.component';
import { FeatureSelectionComponent } from './components/feature-selection/feature-selection.component';
import { ResultComponent } from './components/result/result.component';
import { LoginComponent } from './components/login/login.component';
import { SignupComponent } from './components/signup/signup.component';

import { AuthGuard } from './auth/auth.guard';
import { ResetPasswordComponent } from './components/reset-password/reset-password.component';

const appRoutes: Routes = [
  { path: '', redirectTo: 'signup', pathMatch: 'full' },
  { path: 'signup', component: SignupComponent },
  { path: 'login', component: LoginComponent },
  { path: 'passwordReset', component: ResetPasswordComponent },
  { path: 'home', component: HomeComponent, },
  { path: 'preview', component: PreviewComponent, },
  { path: 'feature-selection', component: FeatureSelectionComponent, },
  { path: 'result', component: ResultComponent, },
];

@NgModule({
  imports: [RouterModule.forRoot(appRoutes, { onSameUrlNavigation: 'reload' })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
