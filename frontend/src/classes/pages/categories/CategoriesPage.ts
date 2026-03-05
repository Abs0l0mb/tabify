'use strict';

import { 
    TitledPage,
    ClientLocation,
    TeamCategoriesTable
} from '@src/classes';

export class CategoriesPage extends TitledPage {

    constructor() {

        super('Categories', 'categories');
        
        const canManageTeams = ClientLocation.get().api.accountData?.access_right_names?.includes('MANAGE TEAMS');
        const urlParam = ClientLocation.get().router.getParams().teamId;
        const myTeamId = ClientLocation.get().api.accountData.team_id;
        const teamId = canManageTeams && typeof urlParam === 'number' ? urlParam : myTeamId;

        this.init(teamId);
    }

    /*
    **
    **
    */
    private init(teamId: number) : void {

        new TeamCategoriesTable(teamId, this.content).addClass('light-zone');
    }
}