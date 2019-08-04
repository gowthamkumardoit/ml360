import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NavigationComponent } from './components/navigation/navigation.component';
import { MaterialModule } from './material.module';
import { AppRoutingModule } from './app-routing.module';
import { HomeComponent } from './components/home/home.component';
import { PreviewComponent } from './components/preview/preview.component';
import { FeatureSelectionComponent } from './components/feature-selection/feature-selection.component';
import { ResultComponent } from './components/result/result.component';
import { LoginComponent } from './components/login/login.component';

import { AngularFireModule } from '@angular/fire';
import { AngularFireAuthModule } from '@angular/fire/auth';
import { AngularFireStorageModule } from '@angular/fire/storage';
import { AngularFirestoreModule } from '@angular//fire/firestore';
import { FlexLayoutModule } from '@angular/flex-layout';

import { environment } from 'src/environments/environment';
import { AuthGuard } from './auth/auth.guard';
import { TableComponent } from './shared/table/table.component';
import { SummaryTableComponent } from './shared/summary-table/summary-table.component';

// Load FusionCharts
import * as FusionCharts from 'fusioncharts';
// Load Charts module
import * as Charts from 'fusioncharts/fusioncharts.charts';
// Load fusion theme
import * as FusionTheme from 'fusioncharts/themes/fusioncharts.theme.fusion';
import * as PowerCharts from 'fusioncharts/fusioncharts.powercharts';

import { FusionChartsModule } from 'angular-fusioncharts';
import { BoxPlotComponent } from './charts/box-plot/box-plot.component';
import { HistogramComponent } from './charts/histogram/histogram.component';
import { BarComponent } from './charts/bar/bar.component';
import { DragDropComponent } from './shared/drag-drop/drag-drop.component';

import { DragDropModule } from '@angular/cdk/drag-drop';
import { MatBottomSheetModule } from '@angular/material/bottom-sheet';
import { BottomSheetComponent } from './shared/bottom-sheet/bottom-sheet.component';
import { SignupComponent } from './components/signup/signup.component';
import { ResetPasswordComponent } from './components/reset-password/reset-password.component';
import { SecureInnerRoutesGuard } from './auth/secure-inner-routes.guard';
import { ConfirmationDialogComponent } from './shared/confirmation-dialog/confirmation-dialog.component';
import { SideNavComponent } from './components/side-nav/side-nav.component';
import { SideNavIconsComponent } from './shared/side-nav-icons/side-nav-icons.component';
import { SettingsComponent } from './components/settings/settings.component';
import { HistoryComponent } from './components/history/history.component';
import { HttpClientModule } from '@angular/common/http';
import { DataTableComponent } from './shared/data-table/data-table.component';
// Add dependencies to FusionChartsModule
FusionChartsModule.fcRoot(FusionCharts, Charts, PowerCharts, FusionTheme);

@NgModule({
  declarations: [
    AppComponent,
    NavigationComponent,
    HomeComponent,
    PreviewComponent,
    FeatureSelectionComponent,
    ResultComponent,
    LoginComponent,
    TableComponent,
    SummaryTableComponent,
    BoxPlotComponent,
    HistogramComponent,
    BarComponent,
    DragDropComponent,
    BottomSheetComponent,
    SignupComponent,
    ResetPasswordComponent,
    ConfirmationDialogComponent,
    SideNavComponent,
    SideNavIconsComponent,
    SettingsComponent,
    HistoryComponent,
    DataTableComponent
  ],
  imports: [
    BrowserModule,
    FusionChartsModule,
    BrowserAnimationsModule,
    MaterialModule,
    AppRoutingModule,
    FormsModule,
    ReactiveFormsModule,
    AngularFireModule.initializeApp(environment.firebaseConfig),
    AngularFireAuthModule,
    AngularFireStorageModule,
    DragDropModule,
    AngularFirestoreModule,
    MatBottomSheetModule,
    FlexLayoutModule,
    HttpClientModule,
  ],
  entryComponents: [
    BottomSheetComponent, ConfirmationDialogComponent
  ],
  providers: [AuthGuard, SecureInnerRoutesGuard],
  bootstrap: [AppComponent]
})
export class AppModule { }
