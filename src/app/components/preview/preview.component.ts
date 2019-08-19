import { Component, OnInit } from '@angular/core';
import { PreviewService } from 'src/app/services/preview.service';
import { AuthService } from 'src/app/services/auth.service';
import { FormControl, Validators } from '@angular/forms';
import { MatDialog, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { SpinnerComponent } from 'src/app/shared/spinner/spinner.component';
import { ActivatedRoute } from '@angular/router';
import { MISSING_NA_CHART } from '../../constant/chart.constants';
import { TOOLTIPS } from '../../constant/app.constants';

@Component({
	selector: 'app-preview',
	templateUrl: './preview.component.html',
	styleUrls: ['./preview.component.scss']
})
export class PreviewComponent implements OnInit {
	rows: any[] = [];
	cols: any[] = [];
	summaryRows: any[] = [];
	summaryCols: any[] = [];
	describeAttributes: any = [];
	describeRows: any = {};
	percentageOfNA: any = {};
	skewAndKurtosis: any = {};
	fileControl = new FormControl('', [Validators.required]);
	filesAvailable;
	isFileSelected: boolean;
	width = 900;
	height = 600;
	type = "scrollcombidy2d";
	dataFormat = "json";
	dataSource = null;
	naChartColumns: any[] = [];
	naCountRows = [];
	naPercentRows = [];
	skewness: any;
	kurtosis: any;
	skewnessAndKurtosis_cols: any;
	resovledData: any;
	tooltips = {};
	constructor(private previewService: PreviewService, private authService: AuthService,
		public dialog: MatDialog, private activatedRoute: ActivatedRoute) {
		this.isFileSelected = false;
	}

	ngOnInit() {
		this.resovledData = this.activatedRoute.snapshot.data;
		if (this.resovledData) {
			this.filesAvailable = this.resovledData.response;
		}
		this.tooltips = TOOLTIPS;
		this.initializePage();
		console.log(this.filesAvailable);
	}

	initializePage() {
		const previousData = JSON.parse(localStorage.getItem('selectedData'));
		if (previousData) {
			// this.callSpinner();
			this.getPreprocessedData(previousData);
		}
	}

	fileSelectionEvent() {

		if (this.fileControl && this.fileControl.value) {
			this.callSpinner();
			localStorage.setItem('selectedFile', JSON.stringify(this.fileControl.value));
			this.previewService.getDownloadURLs(this.fileControl.value).then((response: any) => {
				if (response) {
					localStorage.setItem('selectedData', JSON.stringify(response));
					this.getPreprocessedData(response);
				}
			}).catch((err) => {
				console.log(err);
				this.dialog.closeAll();
			});
		}
	}

	getPreprocessedData(response) {
		if (response) {
			this.naChartColumns = [];
			this.naCountRows = [];
			this.naPercentRows = [];
			this.cols = response.cols;
			this.rows = response.rows;
			this.summaryCols = response.summary_cols;
			this.summaryRows = response.summary_rows;

			response.cols.forEach((elem: any, i: number) => {
				this.naChartColumns.push({ label: elem });
			});

			response.na_data_rows.forEach((elem: any, i: number) => {
				this.naCountRows.push({ value: elem.count_of_missing_values });
				this.naPercentRows.push({ value: elem.percent_of_missing_values });
			});
			this.skewness = response.skew;
			this.kurtosis = response.kurtosis;
			this.skewnessAndKurtosis_cols = Object.keys(response.skew);
			let yMax = response.yMax;
			this.getChart(yMax);
			this.isFileSelected = true;
			this.dialog.closeAll();
		}
	}
	callSpinner() {
		this.dialog.open(SpinnerComponent, { disableClose: true });
	}

	getChart(yMax) {

		const data = {
			chart: {
				...MISSING_NA_CHART,
				pYAxisMaxValue: yMax,
			},
			categories: [
				{
					category: this.naChartColumns
				}
			],
			dataset: [
				{
					seriesname: "Count of NA",
					showvalues: "0",
					plottooltext: "Count of NA in $label : <b>$dataValue</b>",
					data: this.naCountRows
				},

				{
					seriesname: "Percentage of NA",
					parentyaxis: "S",
					renderas: "line",
					showvalues: "0",
					plottooltext: "Percentage of NA in $label : <b>$dataValue</b>%",
					data: this.naPercentRows
				}
			]
		};
		this.dataSource = data;
	}
}


